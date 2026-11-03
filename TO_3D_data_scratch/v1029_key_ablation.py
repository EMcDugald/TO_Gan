"""
v1029_key_ablation.py

Scores candidate bucket keys (coarser TO-spec summaries) on an existing v1029
binned dataset, without regenerating anything. Reads the train and pool .npy
files written by v1029_make_train_data_binned.py (it needs the exact BC points
stored in sample_info for the BC-centroid and load-to-BC-distance fields).

Why: the full v1029 key (shape, load cell, BC cell pattern, VF bin, load
direction) gave 4,103 buckets for 4,450 training parts (85% singletons), so
almost no condition is shared and the label stays redundant. With ~4,450
parts, buckets of 10-20 parts allow only ~200-400 keys, so the key must be
small. Load direction is dropped from every candidate.

Fields available to a key:
  shape      the part's (Nx,Ny,Nz)                          7 values
  load       load cell (k^3)                                8 values at k=2
  vf         VF quantile bin                                4 values
  bccen      cell holding the centroid of the exact BC points  8 values at k=2
  ldist      grid distance (steps) from the load cell to the
             nearest BC cell                                0..3 at k=2
  bcpat      the full BC cell pattern (reference only)      ~91 values
  dir        load-direction class (reference only)          6 values

For each key, on the TRAIN parts: number of buckets, median and mean bucket
size, fraction of parts alone in their bucket and in buckets of >=5, fraction
of parts in buckets holding both labels, the same under label shuffling (same
bucket sizes, labels independent of bucket) and the ratio of the two (near 1:
the bucket says no more about the label than chance; near 0: it fixes it).
On the POOL parts: number with no training part in their bucket, and how
much label information the bucket explains on held-out parts
(info_explained = 1 - LL/LL_base, bucket-majority with smoothing; 0 = none).

Usage:
  python v1029_key_ablation.py --data_dir <binned data folder> [--keys "shape+load+vf,load+vf"]
      [--exclude_shapes 120x20x20]   (prints per-shape positive rates; drops those shapes)
Writes <data_dir>/key_ablation.csv and key_ablation.json.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from v1029_binned_cond_utils import DIR_CLASSES, cell_xyz, point_cell
from v1029_bucket_diagnostics import entropy_bits, auc_score, log_loss

DEFAULT_KEYS = [
    # the two favoured candidates
    "shape+load+vf",
    "shape+load+bccen",
    # neighbours of those
    "shape+load+bccen+vf",
    "shape+load",
    "shape+load+ldist",
    "shape+load+ldist+vf",
    "load+vf",
    "load+bccen",
    "load+bccen+vf",
    # references
    "shape+load+bcpat+vf",
    "shape+load+bcpat+vf+dir",
]


def field_values(info, k):
    shp = tuple(int(v) for v in info["part_shape"])
    bc = np.asarray(info["bc_points_exact"], dtype=np.float64).reshape(-1, 3)
    bccen = point_cell(bc.mean(axis=0), shp, k)
    lx = np.array(cell_xyz(info["load_cell"], k))
    ldist = min(int(np.abs(lx - np.array(cell_xyz(c, k))).sum()) for c in info["bc_cells"])
    dc = info["load_dir_class"]
    return {
        "shape": f"{shp[0]}x{shp[1]}x{shp[2]}",
        "load": f"L{info['load_cell']}",
        "vf": f"V{info['vf_bin']}",
        "bccen": f"C{bccen}",
        "ldist": f"d{ldist}",
        "bcpat": "B" + "-".join(str(c) for c in info["bc_cells"]),
        "dir": "D" + (DIR_CLASSES[dc] if dc >= 0 else "zero"),
    }


def load_split(path, k, exclude=()):
    data = np.load(path, allow_pickle=True)
    fv, y = [], []
    for d in data:
        info = d[4]
        s = "x".join(str(int(v)) for v in info["part_shape"])
        if s in exclude:
            continue
        fv.append(field_values(info, k))
        y.append(1.0 if info["manuf_label"] > 0 else 0.0)
    return fv, np.array(y)


def score_key(fields, fv_tr, y_tr, fv_po, y_po, n_shuffles, smoothing, rng):
    key = lambda f: "|".join(f[x] for x in fields)
    k_tr = [key(f) for f in fv_tr]
    k_po = [key(f) for f in fv_po]
    ids = {kk: j for j, kk in enumerate(sorted(set(k_tr)))}
    kid = np.array([ids[kk] for kk in k_tr])
    nb = len(ids)
    ntot = np.bincount(kid, minlength=nb).astype(np.float64)
    npos = np.bincount(kid, weights=y_tr, minlength=nb)
    size_pp = ntot[kid]
    mixed_pp = ((npos > 0) & (npos < ntot))[kid]
    base = float(y_tr.mean())
    H = float(entropy_bits(base))
    Hr = float(np.sum(ntot * entropy_bits(npos / ntot)) / len(y_tr)) / H

    nm, nh = [], []
    for _ in range(n_shuffles):
        ys = rng.permutation(y_tr)
        p = np.bincount(kid, weights=ys, minlength=nb)
        nm.append(float((((p > 0) & (p < ntot))[kid]).mean()))
        nh.append(float(np.sum(ntot * entropy_bits(p / ntot)) / len(ys)) / H)
    nm, nh = float(np.mean(nm)), float(np.mean(nh))

    p_po = np.array([(npos[ids[kk]] + smoothing * base) / (ntot[ids[kk]] + smoothing)
                     if kk in ids else base for kk in k_po])
    ll = log_loss(y_po, p_po)
    ll0 = log_loss(y_po, np.full(len(y_po), base))
    return {
        "key": "+".join(fields),
        "n_buckets": nb,
        "median_size": float(np.median(ntot)),
        "mean_size": float(ntot.mean()),
        "frac_singleton": float(np.mean(size_pp == 1)),
        "frac_in_ge5": float(np.mean(size_pp >= 5)),
        "frac_mixed": float(mixed_pp.mean()),
        "frac_mixed_shuffled": nm,
        "mixing_ratio": float(mixed_pp.mean() / nm) if nm > 0 else float("nan"),
        "H_ratio": Hr,
        "H_ratio_shuffled": nh,
        "pool_orphans": int(sum(1 for kk in k_po if kk not in ids)),
        "pool_n": len(k_po),
        "heldout_info_explained": float(1.0 - ll / ll0),
        "heldout_auc": auc_score(y_po, p_po),
    }


def main():
    ap = argparse.ArgumentParser(description="Score candidate bucket keys on v1029 binned data.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--keys", default="", help="comma-separated keys, fields joined by '+'")
    ap.add_argument("--n_shuffles", type=int, default=200)
    ap.add_argument("--exclude_shapes", default="",
                    help="comma-separated shapes to leave out, e.g. 120x20x20")
    ap.add_argument("--smoothing", type=float, default=1.0)
    args = ap.parse_args()

    tr = sorted(glob.glob(os.path.join(args.data_dir, "*_train.npy")))
    po = sorted(glob.glob(os.path.join(args.data_dir, "*_pool.npy")))
    if len(tr) != 1 or len(po) != 1:
        raise SystemExit(f"Expected one *_train.npy and one *_pool.npy in {args.data_dir}")
    k = int(np.load(tr[0][:-4] + "_meta.npz", allow_pickle=True)["k_bins"])
    print("Loading train/pool ...")
    exclude = set(t.strip() for t in args.exclude_shapes.split(",") if t.strip())
    fv_tr, y_tr = load_split(tr[0], k, exclude)
    fv_po, y_po = load_split(po[0], k, exclude)
    if exclude:
        print(f"Excluding shapes: {sorted(exclude)}")
    print(f"Train {len(y_tr)} ({100*y_tr.mean():.1f}% positive, {int((1-y_tr).sum())} negative), "
          f"pool {len(y_po)}, k={k}")
    print("Per-shape training labels (n, % positive, n negative):")
    by = defaultdict(list)
    for d, yy in zip(fv_tr, y_tr):
        by[d["shape"]].append(yy)
    for s_ in sorted(by, key=lambda t: np.mean(by[t])):
        a = np.array(by[s_])
        print(f"  {s_:>10}: {len(a):5d}  {100*a.mean():5.1f}%  {int((1-a).sum()):5d}")

    for f in ["bccen", "ldist"]:
        vals = defaultdict(int)
        for d in fv_tr:
            vals[d[f]] += 1
        print(f"  {f}: " + ", ".join(f"{v}={n}" for v, n in sorted(vals.items())))

    keys = [s.strip() for s in args.keys.split(",") if s.strip()] or DEFAULT_KEYS
    rng = np.random.default_rng(0)
    rows = []
    for ks in keys:
        fields = ks.split("+")
        bad = [f for f in fields if f not in fv_tr[0]]
        if bad:
            raise SystemExit(f"Unknown field(s) {bad} in key {ks!r}")
        rows.append(score_key(fields, fv_tr, y_tr, fv_po, y_po, args.n_shuffles, args.smoothing, rng))

    hdr = (f"\n{'key':26s} {'buckets':>7s} {'median':>6s} {'single':>6s} {'>=5':>5s} "
           f"{'mixed':>6s} {'shuf':>6s} {'ratio':>5s} {'orphan':>6s} {'info':>6s} {'AUC':>5s}")
    print(hdr)
    for r in rows:
        print(f"{r['key']:26s} {r['n_buckets']:7d} {r['median_size']:6.0f} "
              f"{100*r['frac_singleton']:5.1f}% {100*r['frac_in_ge5']:4.0f}% "
              f"{100*r['frac_mixed']:5.1f}% {100*r['frac_mixed_shuffled']:5.1f}% "
              f"{r['mixing_ratio']:5.2f} {r['pool_orphans']:6d} "
              f"{r['heldout_info_explained']:6.3f} {r['heldout_auc']:5.3f}")
    print("\nColumns: buckets/median = train buckets and median parts per bucket; single, >=5 = "
          "% of train parts alone / in buckets of 5+; mixed vs shuf = % of train parts in "
          "buckets holding both labels, observed vs label-shuffled; ratio = mixed/shuf (near 1 "
          "good); orphan = pool conditions with no train neighbor; info = held-out label "
          "information the bucket explains (near 0 good); AUC = same as an AUC.")

    with open(os.path.join(args.data_dir, "key_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.data_dir, "key_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {os.path.join(args.data_dir, 'key_ablation.csv')} and .json")


if __name__ == "__main__":
    main()
