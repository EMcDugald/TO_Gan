"""
v1029_bucket_diagnostics.py

Go/no-go check for the binned-condition dataset BEFORE spending GPU time:
does the manufacturability label carry information that the binned TO spec
does not?

Two questions, answered separately:

1. Training signal (the primary one). The model can only learn to use the
   label from training conditions that occur with BOTH labels. With exact
   coordinates every condition was unique, so this fraction was 0 and the
   label was ignorable. Reported: fraction of training parts whose bucket
   holds both labels, and how much label entropy is left once the bucket is
   known, H(label | bucket) / H(label). Near 0 means the bucket fixes the
   label (bad); near 1 means the bucket says little about it (good).

2. Held-out predictability (secondary). Classifiers predict the label of
   POOL parts from their condition alone, trained on the TRAIN parts:
     bucket-majority (smoothed per-bucket positive rate, nonparametric),
     logistic regression on the binned condition,
     logistic regression on EXACT spec features (what the v0804 models saw,
     summarized), and, if scikit-learn is installed, gradient boosting on
     both feature sets.
   Each is scored by AUC and by "information explained" = 1 - LL / LL_base,
   where LL is pool log-loss and LL_base the log-loss of always predicting
   the training base rate. 0 = the condition says nothing about the label,
   1 = the condition determines it.

Interpretation printed at the end flags (does not block) weak cases.

Usage:
  python v1029_bucket_diagnostics.py --data_dir <outdir of v1029_make_train_data_binned.py>
Writes <data_dir>/bucket_diagnostics.json.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


# -------------------------------------------------------------------------
# metrics
# -------------------------------------------------------------------------

def entropy_bits(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def auc_score(y, s):
    """y in {0,1}; ties get average rank (Mann-Whitney)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(name, y_pool, p_pool, base_rate):
    ll = log_loss(y_pool, p_pool)
    ll0 = log_loss(y_pool, np.full(len(y_pool), base_rate))
    acc = float(np.mean((p_pool >= 0.5) == (y_pool > 0.5)))
    return {"model": name, "auc": auc_score(y_pool, p_pool), "log_loss": ll,
            "log_loss_base": ll0, "info_explained": float(1.0 - ll / ll0),
            "accuracy": acc, "accuracy_base": float(max(np.mean(y_pool), 1 - np.mean(y_pool)))}


# -------------------------------------------------------------------------
# numpy logistic regression (Newton / IRLS with L2), no sklearn needed
# -------------------------------------------------------------------------

def fit_logreg(X, y, l2=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R + 1e-8 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return lambda Xn: 1.0 / (1.0 + np.exp(-np.clip(
        np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ w, -30, 30)))


# -------------------------------------------------------------------------
# features
# -------------------------------------------------------------------------

def load_split(path):
    data = np.load(path, allow_pickle=True)
    return [(d[1], d[4]) for d in data]


def rel_pos(coord01, shape):
    shape = np.asarray(shape, dtype=np.float64)
    c = float(shape.max())
    idx = np.clip(np.rint(np.asarray(coord01, dtype=np.float64) * (c - 1.0)), 0, shape - 1)
    return (idx + 0.5) / shape


def features(entries, slices, shape_list):
    a = slices["manuf_label"][0]
    Xb, Xe, y, keys, shp = [], [], [], [], []
    for cond, info in entries:
        s = tuple(int(v) for v in info["part_shape"])
        onehot = np.array([1.0 if s == t else 0.0 for t in shape_list])
        Xb.append(np.concatenate([np.asarray(cond[:a], dtype=np.float64), onehot]))
        bc = rel_pos(np.asarray(info["bc_points_exact"]).reshape(-1, 3), s)
        Xe.append(np.concatenate([
            onehot, rel_pos(info["load_point_exact"], s),
            np.asarray(info["load_vec_exact"], dtype=np.float64),
            [info["volume_fraction"], len(bc)], bc.min(0), bc.max(0), bc.mean(0)]))
        y.append(1.0 if info["manuf_label"] > 0 else 0.0)
        keys.append(info["bucket_key"])
        shp.append(s)
    return np.array(Xb), np.array(Xe), np.array(y), keys, shp


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Go/no-go label diagnostics for v1029 binned data.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="pseudo-count pulling small buckets toward the base rate")
    ap.add_argument("--n_shuffles", type=int, default=200)
    args = ap.parse_args()

    tr_path = sorted(glob.glob(os.path.join(args.data_dir, "*_train.npy")))
    po_path = sorted(glob.glob(os.path.join(args.data_dir, "*_pool.npy")))
    if len(tr_path) != 1 or len(po_path) != 1:
        raise SystemExit(f"Expected exactly one *_train.npy and one *_pool.npy in {args.data_dir}")
    meta = np.load(tr_path[0][:-4] + "_meta.npz", allow_pickle=True)
    slices = {k: tuple(v) for k, v in json.loads(str(meta["cond_slices_json"])).items()}
    train, pool = load_split(tr_path[0]), load_split(po_path[0])
    shape_list = sorted(set(tuple(int(v) for v in info["part_shape"]) for _, info in train + pool))
    Xb_tr, Xe_tr, y_tr, k_tr, s_tr = features(train, slices, shape_list)
    Xb_po, Xe_po, y_po, k_po, s_po = features(pool, slices, shape_list)
    base = float(y_tr.mean())
    print(f"Train {len(y_tr)} parts ({100*base:.1f}% positive)   Pool {len(y_po)} parts "
          f"({100*y_po.mean():.1f}% positive)")

    # ---------- 1. training signal ----------
    cnt = defaultdict(lambda: [0, 0])
    for kk, yy in zip(k_tr, y_tr):
        cnt[kk][int(yy)] += 1
    n_b = np.array([sum(v) for v in cnt.values()])
    sizes_per_part = np.array([sum(cnt[kk]) for kk in k_tr])
    mixed_part = np.array([cnt[kk][0] > 0 and cnt[kk][1] > 0 for kk in k_tr])
    H = float(entropy_bits(base))
    # plug-in conditional entropy, weighted by bucket size (biased low for tiny buckets)
    Hc = float(sum(sum(v) * entropy_bits(v[1] / sum(v)) for v in cnt.values()) / len(y_tr))
    big = {kk: v for kk, v in cnt.items() if sum(v) >= 5}
    nbig = sum(sum(v) for v in big.values())
    Hc_big = (float(sum(sum(v) * entropy_bits(v[1] / sum(v)) for v in big.values()) / nbig)
              if nbig else float("nan"))
    Hb_big = (float(entropy_bits(sum(v[1] for v in big.values()) / nbig)) if nbig else float("nan"))

    print("\n--- 1. Training signal: do shared conditions hold both labels? ---")
    print(f"Distinct train buckets: {len(n_b)}   median size {np.median(n_b):.0f}   "
          f"mean {n_b.mean():.1f}   max {n_b.max()}")
    print(f"Train parts in singleton buckets: {100*np.mean(sizes_per_part == 1):.1f}%   "
          f"in buckets of >=5: {100*np.mean(sizes_per_part >= 5):.1f}%")
    print(f"Train parts in buckets holding BOTH labels: {100*mixed_part.mean():.1f}%")
    print(f"Label entropy left once the bucket is known: H(label|bucket)/H(label) = "
          f"{Hc/H:.2f}  (all buckets; singletons count as 0, so this is a lower bound)")
    print(f"  same ratio restricted to buckets of >=5 parts ({nbig} parts): {Hc_big/Hb_big:.2f}")
    # Label-shuffle baseline: what mixing and entropy would look like if labels
    # were independent of the bucket, with these exact bucket sizes. Small
    # buckets cap the mixed fraction even for random labels, so the observed
    # numbers are read against this baseline, not against 100%.
    rng = np.random.default_rng(0)
    key_ids = {kk: j for j, kk in enumerate(cnt)}
    kid = np.array([key_ids[kk] for kk in k_tr])
    null_mixed, null_H = [], []
    for _ in range(args.n_shuffles):
        ys = rng.permutation(y_tr)
        npos = np.bincount(kid, weights=ys, minlength=len(key_ids))
        ntot = np.bincount(kid, minlength=len(key_ids)).astype(np.float64)
        mixed_b = (npos > 0) & (npos < ntot)
        null_mixed.append(float(mixed_b[kid].mean()))
        null_H.append(float(np.sum(ntot * entropy_bits(npos / ntot)) / len(ys)) / H)
    null_mixed_m, null_H_m = float(np.mean(null_mixed)), float(np.mean(null_H))
    print(f"Label-shuffle baseline (same bucket sizes, labels independent of bucket, "
          f"{args.n_shuffles} shuffles):")
    print(f"  parts in mixed-label buckets: observed {100*mixed_part.mean():.1f}%  vs  "
          f"shuffled {100*null_mixed_m:.1f}%   -> ratio {mixed_part.mean()/max(null_mixed_m,1e-9):.2f}")
    print(f"  entropy ratio:                observed {Hc/H:.2f}   vs  shuffled {null_H_m:.2f}   "
          f"-> ratio {(Hc/H)/max(null_H_m,1e-9):.2f}")
    print("  (ratio near 1: the bucket says no more about the label than chance; near 0: the "
          "bucket fixes the label)")
    per_shape = {}
    for s in shape_list:
        m = np.array([t == s for t in s_tr])
        if m.any():
            per_shape[f"{s[0]}x{s[1]}x{s[2]}"] = {
                "n_train": int(m.sum()),
                "n_buckets": int(len(set(np.array(k_tr, dtype=object)[m]))),
                "frac_in_mixed_buckets": float(mixed_part[m].mean())}
    print("Per shape (n_train, buckets, % in mixed-label buckets):")
    for s, v in per_shape.items():
        print(f"  {s:>10}: {v['n_train']:5d}  {v['n_buckets']:5d}  {100*v['frac_in_mixed_buckets']:5.1f}%")

    # ---------- 2. held-out predictability ----------
    a = args.smoothing
    p_bucket = np.array([(cnt[kk][1] + a * base) / (sum(cnt[kk]) + a) if kk in cnt else base
                         for kk in k_po])
    results = [score("bucket-majority (smoothed)", y_po, p_bucket, base)]
    results.append(score("logistic, binned condition", y_po, fit_logreg(Xb_tr, y_tr)(Xb_po), base))
    results.append(score("logistic, exact spec features", y_po, fit_logreg(Xe_tr, y_tr)(Xe_po), base))
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        for name, Xt, Xp in [("grad. boosting, binned condition", Xb_tr, Xb_po),
                             ("grad. boosting, exact spec features", Xe_tr, Xe_po)]:
            m = HGB(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0)
            m.fit(Xt, y_tr)
            results.append(score(name, y_po, m.predict_proba(Xp)[:, 1], base))
    except ImportError:
        print("(scikit-learn not installed: gradient-boosting rows skipped)")

    print("\n--- 2. Held-out predictability: label of POOL parts from condition alone ---")
    print(f"{'model':38s} {'AUC':>6s} {'info expl.':>10s} {'acc':>6s} {'acc base':>8s}")
    for r in results:
        print(f"{r['model']:38s} {r['auc']:6.3f} {r['info_explained']:10.3f} "
              f"{r['accuracy']:6.3f} {r['accuracy_base']:8.3f}")

    # ---------- verdict (flags, not blocks) ----------
    print("\n--- Reading ---")
    flags = []
    mix_ratio = mixed_part.mean() / max(null_mixed_m, 1e-9)
    if mix_ratio < 0.5:
        flags.append(f"FLAG: label mixing is only {mix_ratio:.2f}x what random labels would give "
                     "with these bucket sizes; the bucket largely fixes the label, so the model "
                     "may still ignore it. Coarsen the bins or drop a field.")
    if mixed_part.mean() < 0.2:
        flags.append(f"FLAG: only {100*mixed_part.mean():.1f}% of training parts share a "
                     "condition with a part of the other label, so the direct training signal "
                     "for using the label is small (check the baseline line to see whether "
                     "sparse buckets or redundancy is the cause).")
    binned_info = max(r["info_explained"] for r in results if "binned" in r["model"] or "bucket" in r["model"])
    if binned_info > 0.5:
        flags.append("FLAG: the binned condition explains more than half the label information on "
                     "held-out parts; the label is still largely redundant.")
    if np.mean(sizes_per_part == 1) > 0.3:
        flags.append("FLAG: over 30% of training parts sit alone in their bucket; ground-truth "
                     "neighborhoods will often be empty or tiny.")
    for f in flags:
        print(f)
    if not flags:
        print("No flags: shared conditions routinely hold both labels and the binned condition "
              "leaves most label information unexplained, so the label is real information the "
              "model has to use.")

    out = {"train_file": tr_path[0], "pool_file": po_path[0], "base_rate_train": base,
           "n_train_buckets": int(len(n_b)), "bucket_size_median": float(np.median(n_b)),
           "frac_train_parts_singleton": float(np.mean(sizes_per_part == 1)),
           "frac_train_parts_in_mixed_buckets": float(mixed_part.mean()),
           "H_label_given_bucket_over_H_label": Hc / H,
           "H_ratio_buckets_ge5": Hc_big / Hb_big if nbig else None,
           "shuffle_baseline_frac_mixed": null_mixed_m,
           "shuffle_baseline_H_ratio": null_H_m,
           "mixing_ratio_vs_shuffle": float(mix_ratio),
           "per_shape": per_shape, "heldout": results, "flags": flags}
    with open(os.path.join(args.data_dir, "bucket_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.join(args.data_dir, 'bucket_diagnostics.json')}")


if __name__ == "__main__":
    main()
