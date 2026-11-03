"""
v1029_key_selection_report.py

Builds a shareable report explaining how the coarse condition ("key") for the
v1029b manufacturability model was chosen. Everything is recomputed from the
data files, so the numbers in the report are the real ones.

Inputs
  --binned_dir   folder from v1029_make_train_data_binned.sh (full binned key;
                 holds *_train.npy, *_pool.npy, bucket_diagnostics.json)
  --keyed_dir    optional folder from v1029b_make_train_data_keyed.sh; if it
                 has viz_train/buckets/*.png, a few galleries are included
  --chosen_key   the key that was selected (default shape+load+vf)

Outputs (default <binned_dir>/key_selection_report/)
  report.html                 self-contained page (figures embedded), to share
  report.json                 every number shown in the report
  fig1_bucket_sizes.png       neighborhood sizes, full key vs chosen key
  fig2_field_cardinality.png  distinct values per field and why the full key explodes
  fig3_key_ablation.png       candidate keys: neighborhood size vs label mixing,
                              and held-out label information per key
  fig4_per_shape_labels.png   positive rate and negatives per shape
  fig5_label_mix_chosen.png   label mix per bucket under the chosen key
  gallery_*.png               copied bucket galleries (if --keyed_dir given)

Usage (in the to_gan environment, from TO_3D_data_scratch):
  python v1029_key_selection_report.py \
      --binned_dir /xdisk/hdb/emcdugald/train_data/diffusion_binned/binned_k2_vf4_p80_split0 \
      --keyed_dir  /xdisk/hdb/emcdugald/train_data/diffusion_binned/keyed_load-vf_k2_vf4_p80_split0
"""

import argparse
import base64
import glob
import html
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v1029_key_ablation import load_split, score_key

FULL_KEY = "shape+load+bcpat+vf+dir"
CANDIDATES = [
    FULL_KEY,
    "shape+load+bcpat+vf",
    "shape+load+bccen+vf",
    "shape+load+ldist+vf",
    "shape+load+bccen",
    "shape+load+ldist",
    "shape+load+vf",
    "shape+load",
    "load+vf",
    "load+bccen",
]
FIELD_LABELS = {"shape": "part shape", "load": "load octant", "bcpat": "BC octant pattern",
                "vf": "VF quartile", "dir": "load direction", "bccen": "BC centroid octant",
                "ldist": "load-to-BC distance"}


def bucket_stats(fv, y, fields):
    by = defaultdict(list)
    for f, yy in zip(fv, y):
        by["|".join(f[x] for x in fields)].append(yy)
    sizes = np.array([len(v) for v in by.values()])
    fpos = np.array([np.mean(v) for v in by.values()])
    return sizes, fpos


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Report on how the v1029b coarse key was chosen.")
    ap.add_argument("--binned_dir", required=True)
    ap.add_argument("--keyed_dir", default="")
    ap.add_argument("--chosen_key", default="shape+load+vf")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--n_shuffles", type=int, default=200)
    args = ap.parse_args()

    out = args.outdir or os.path.join(args.binned_dir, "key_selection_report")
    os.makedirs(out, exist_ok=True)
    tr = sorted(glob.glob(os.path.join(args.binned_dir, "*_train.npy")))
    po = sorted(glob.glob(os.path.join(args.binned_dir, "*_pool.npy")))
    if len(tr) != 1 or len(po) != 1:
        raise SystemExit(f"Expected one *_train.npy and one *_pool.npy in {args.binned_dir}")
    meta = np.load(tr[0][:-4] + "_meta.npz", allow_pickle=True)
    k = int(meta["k_bins"])
    thr = float(meta["manuf_threshold_value"])
    pct = float(meta["manuf_percentile_threshold"])
    print("Loading train/pool (about a minute) ...")
    fv_tr, y_tr = load_split(tr[0], k)
    fv_po, y_po = load_split(po[0], k)
    n_tr, n_po = len(y_tr), len(y_po)
    chosen = args.chosen_key
    cands = list(CANDIDATES)
    if chosen not in cands:
        cands.append(chosen)

    # ---------- candidate keys ----------
    rng = np.random.default_rng(0)
    rows = [score_key(c.split("+"), fv_tr, y_tr, fv_po, y_po, args.n_shuffles, 1.0, rng)
            for c in cands]
    R = {r["key"]: r for r in rows}

    # ---------- field cardinality ----------
    card = {f: len(set(d[f] for d in fv_tr)) for f in ["shape", "load", "bcpat", "vf", "dir"]}
    product = int(np.prod(list(card.values())))

    # ---------- per shape ----------
    by_shape = defaultdict(list)
    for d, yy in zip(fv_tr, y_tr):
        by_shape[d["shape"]].append(yy)
    per_shape = sorted(
        ({"shape": s, "n": len(v), "frac_positive": float(np.mean(v)),
          "n_negative": int(len(v) - np.sum(v))} for s, v in by_shape.items()),
        key=lambda r: r["frac_positive"])

    diag_path = os.path.join(args.binned_dir, "bucket_diagnostics.json")
    diag = json.load(open(diag_path)) if os.path.exists(diag_path) else None

    # ---------- figures ----------
    full_sizes, _ = bucket_stats(fv_tr, y_tr, FULL_KEY.split("+"))
    ch_sizes, ch_fpos = bucket_stats(fv_tr, y_tr, chosen.split("+"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, sizes, name in [(axs[0], full_sizes, f"full binned key\n({FULL_KEY})"),
                            (axs[1], ch_sizes, f"chosen key\n({chosen})")]:
        bins = np.arange(1, sizes.max() + 2) - 0.5 if sizes.max() <= 60 else 40
        ax.hist(sizes, bins=bins, color="0.35")
        ax.set_xlabel("training parts sharing one condition (bucket size)")
        ax.set_ylabel("number of conditions")
        ax.set_title(f"{name}\n{len(sizes)} conditions, median {np.median(sizes):.0f} parts each",
                     fontsize=10)
    savefig(fig, os.path.join(out, "fig1_bucket_sizes.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [FIELD_LABELS[f] for f in card]
    ax.bar(names, list(card.values()), color=["0.55", "tab:red", "tab:blue", "tab:green", "0.55"])
    for i, v in enumerate(card.values()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("distinct values in training data")
    ax.set_title(f"Possible full-key combinations: {' x '.join(map(str, card.values()))} = "
                 f"{product:,}  vs  {n_tr:,} training parts", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=15)
    savefig(fig, os.path.join(out, "fig2_field_cardinality.png"))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axs[0]
    for i, r in enumerate(rows):
        has_shape = r["key"].startswith("shape")
        c = "tab:orange" if r["key"] == chosen else ("0.3" if has_shape else "0.7")
        ax.scatter(r["median_size"], r["mixing_ratio"], s=40 + 400 * max(r["heldout_info_explained"], 0),
                   color=c, edgecolors="k", zorder=3, marker="o" if has_shape else "s")
        ax.annotate(r["key"], (r["median_size"], r["mixing_ratio"]), fontsize=7,
                    xytext=(6, -10 + 10 * (i % 3)), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("median training parts per condition (log)")
    ax.set_ylabel("label mixing vs shuffled labels (1 = label independent of condition)")
    ax.set_title("Candidate keys (marker size = held-out label info explained;\n"
                 "squares = keys without shape, not achievable: shape reaches the model via the mask)",
                 fontsize=9)
    ax = axs[1]
    order = sorted(rows, key=lambda r: r["heldout_info_explained"])
    ax.barh([r["key"] for r in order], [r["heldout_info_explained"] for r in order],
            color=["tab:orange" if r["key"] == chosen else "0.5" for r in order])
    if "shape+load" in R:
        ax.axvline(R["shape+load"]["heldout_info_explained"], color="tab:red", ls="--", lw=1,
                   label="shape + load only (approx. floor set by the shape mask)")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("held-out label information explained by the condition (0 = none)")
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, os.path.join(out, "fig3_key_ablation.png"))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(per_shape))
    ax1.bar(xs, [100 * r["frac_positive"] for r in per_shape], color="tab:green", alpha=0.8)
    ax1.axhline(100 * y_tr.mean(), color="k", ls="--", lw=0.8, label="overall")
    ax1.set_ylabel("% positive"); ax1.set_ylim(0, 100)
    ax1.set_xticks(xs); ax1.set_xticklabels([r["shape"] for r in per_shape])
    for i, r in enumerate(per_shape):
        ax1.text(i, 100 * r["frac_positive"] + 1, f"{r['n_negative']} neg\nof {r['n']}",
                 ha="center", fontsize=7)
    ax1.legend(fontsize=8)
    ax1.set_title(f"Training labels by shape (positive = deformation_p99 <= {thr:.4g}, "
                  f"the p{pct:g} of all labeled parts)", fontsize=10)
    savefig(fig, os.path.join(out, "fig4_per_shape_labels.png"))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    jit = np.random.default_rng(0).uniform(-0.2, 0.2, len(ch_sizes))
    ax.scatter(ch_sizes + jit, ch_fpos, s=12, alpha=0.6)
    ax.axhline(y_tr.mean(), color="k", ls="--", lw=0.8, label=f"base rate {y_tr.mean():.2f}")
    ax.set_xlabel("training parts in condition"); ax.set_ylabel("fraction positive")
    ax.set_title(f"Chosen key: label mix in each of {len(ch_sizes)} conditions\n"
                 "(points between 0 and 1 = the same condition has both labels)", fontsize=10)
    ax.legend(fontsize=8)
    savefig(fig, os.path.join(out, "fig5_label_mix_chosen.png"))

    galleries = []
    if args.keyed_dir:
        for j, g in enumerate(sorted(glob.glob(os.path.join(args.keyed_dir, "viz_train", "buckets",
                                                            "bucket_*.png")))[:3]):
            dst = os.path.join(out, f"gallery_{j}.png")
            shutil.copy(g, dst)
            galleries.append(dst)

    # ---------- json ----------
    report = {
        "train_file": tr[0], "pool_file": po[0], "n_train": n_tr, "n_pool": n_po,
        "frac_positive_train": float(y_tr.mean()), "manuf_threshold_value": thr,
        "manuf_percentile_threshold": pct, "k_bins": k,
        "full_key": FULL_KEY, "chosen_key": chosen,
        "field_cardinality": card, "full_key_combinations": product,
        "candidate_keys": rows, "per_shape": per_shape,
        "full_key_diagnostics": diag,
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------- html ----------
    def img(path, width="100%"):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:{width}">'

    F, C = R[FULL_KEY], R[chosen]
    sl = R.get("shape+load")
    lv = R.get("load+vf")
    held = ""
    if diag:
        hrows = "".join(
            f"<tr><td>{html.escape(h['model'])}</td><td>{h['auc']:.3f}</td>"
            f"<td>{h['info_explained']:.3f}</td></tr>" for h in diag["heldout"])
        ex_best = max(0.0, max(h["info_explained"] for h in diag["heldout"] if "exact" in h["model"]))
        bn_best = max(0.0, max(h["info_explained"] for h in diag["heldout"] if "exact" not in h["model"]))
        held = (f"<p>For the full binned key, classifiers trained on the training parts predicted "
                f"the label of held-out parts from the condition alone as follows. The best binned "
                f"classifier explains {bn_best:.2f} of the label information against {ex_best:.2f} "
                f"for the exact specification, so binning lost little of what the specification "
                f"says about the label. The remaining {100*(1-ex_best):.0f}% is not in the "
                f"specification at all: it lives in the details of each part, which is what the "
                f"label can add.</p>"
                f"<table><tr><th>classifier</th><th>AUC</th><th>info explained</th></tr>{hrows}</table>")

    def krow(r):
        cls = ' class="chosen"' if r["key"] == chosen else ""
        return (f"<tr{cls}><td>{html.escape(r['key'])}</td><td>{r['n_buckets']}</td>"
                f"<td>{r['median_size']:.0f}</td><td>{100*r['frac_singleton']:.1f}%</td>"
                f"<td>{100*r['frac_mixed']:.1f}%</td><td>{100*r['frac_mixed_shuffled']:.1f}%</td>"
                f"<td>{r['mixing_ratio']:.2f}</td><td>{r['pool_orphans']}</td>"
                f"<td>{r['heldout_info_explained']:.3f}</td></tr>")

    shape_floor = (f"Shape plus load octant alone already explains {sl['heldout_info_explained']:.2f} "
                   f"of the held-out label information, while load octant plus VF without shape "
                   f"explains only {lv['heldout_info_explained']:.2f}." if sl and lv else "")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Choosing the coarse condition</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 980px;
       margin: 32px auto; padding: 0 16px; line-height: 1.5; color: #222; background: #fff; }}
h1 {{ font-size: 1.6em; }} h2 {{ margin-top: 1.8em; border-bottom: 1px solid #ddd; }}
table {{ border-collapse: collapse; font-size: 0.85em; margin: 0.8em 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
tr.chosen {{ background: #fff1d6; font-weight: 600; }}
.note {{ color: #555; font-size: 0.9em; }}
</style></head><body>
<h1>Choosing a coarse condition for the manufacturability-conditioned generator</h1>
<p class="note">Data: {n_tr:,} training and {n_po:,} held-out (pool) parts with Ansys labels,
{100*y_tr.mean():.0f}% positive (positive = deformation_p99 &le; {thr:.4g}, the {pct:g}th
percentile). Octant grid k = {k}. All numbers recomputed from the data files.</p>

<h2>1. The problem</h2>
<p>The earlier models were conditioned on the exact TO specification (BC points and DOFs, load
point and direction, volume fraction, shape) plus a manufacturability label. Asking them for
positive or negative parts gave the same result, about 80% positive, which is the base rate
(that figure is from the earlier surrogate check). The reason: in the data, each exact
specification appears with exactly one part, so the label never distinguishes two parts that
share a condition. The model can predict every part from the specification alone and learns to
ignore the label.</p>
<p>The fix is to describe the specification coarsely, so that many parts share one condition.
Within a shared condition some parts are positive and some negative, and only the label tells
them apart, so the model has a reason to use it. The same shared conditions also give natural
groups of real parts (&ldquo;neighborhoods&rdquo;) to compare generated parts against.</p>

<h2>2. How each candidate was measured</h2>
<p><b>Conditions (buckets).</b> Parts with the same coarse description form one bucket. We want
buckets large enough that many hold both labels and that every held-out condition has real
neighbors.</p>
<p><b>Label mixing vs shuffled labels.</b> The share of training parts whose bucket holds both
labels, divided by the same share after randomly shuffling labels (same bucket sizes). Near 1
means the condition says no more about the label than chance, which is what we want; near 0 means
the condition fixes the label, the original problem. Shuffling matters because tiny buckets
cannot be mixed even when labels are random.</p>
<p><b>Held-out label information.</b> Using only training parts, predict the label of each
held-out part from its condition (per-bucket positive rate), and measure how much better that
is than always guessing the base rate (1 &minus; log-loss / base log-loss). 0 means the condition
carries no label information; 1 means it determines the label.</p>

<h2>3. First attempt: every field binned into octants</h2>
<p>Load point and BC points were replaced by the octants of the part that contain them, VF by
its quartile, load direction by its dominant axis and sign. This still left
{F['n_buckets']:,} distinct conditions for {n_tr:,} parts: median bucket size
{F['median_size']:.0f}, {100*F['frac_singleton']:.0f}% of parts alone in their bucket, only
{100*F['frac_mixed']:.1f}% in a bucket with both labels, and {F['pool_orphans']} of {n_po}
held-out conditions with no training neighbor at all.</p>
{img(os.path.join(out, "fig1_bucket_sizes.png"))}
{held}
<p>The cause is the number of combinations. Every field is spread fairly evenly over its values,
so the number of possible conditions is roughly the product of the field sizes, about
{product:,} here. NITO supports are a few scattered patches, so the BC octant pattern alone takes
{card['bcpat']} distinct values. With {n_tr:,} parts, neighborhoods of 10&ndash;20 parts allow
only a few hundred conditions in total.</p>
{img(os.path.join(out, "fig2_field_cardinality.png"), "80%")}

<h2>4. Comparing smaller keys</h2>
<p>Load direction was dropped from all candidates. BC information was offered in compact forms:
the octant holding the BC centroid, or the grid distance from the load octant to the nearest BC
octant.</p>
<table><tr><th>key</th><th>conditions</th><th>median size</th><th>alone</th><th>mixed</th>
<th>mixed (shuffled)</th><th>mixing ratio</th><th>held-out conditions without neighbors</th>
<th>held-out label info</th></tr>
{''.join(krow(r) for r in rows)}</table>
{img(os.path.join(out, "fig3_key_ablation.png"))}
<p><b>Shape sets a floor.</b> {shape_floor} Some shapes are simply more often negative (below).
The part's shape always reaches the model through its domain mask, whatever the condition
vector contains, so keys without shape (squares in the figure) are not achievable, and about
{sl['heldout_info_explained'] if sl else float('nan'):.2f} of label information is explained by
shape no matter which key is used. Measured against that floor, VF and the BC summaries add
little.</p>
{img(os.path.join(out, "fig4_per_shape_labels.png"), "85%")}

<h2>5. Decision: {html.escape(chosen)}</h2>
<p>The condition is the load octant (8 values) and the VF quartile (4 values), plus the
manufacturability label; shape comes through the mask. This gives {C['n_buckets']} conditions with
a median of {C['median_size']:.0f} parts, {100*C['frac_singleton']:.1f}% singletons,
{100*C['frac_mixed']:.1f}% of parts in buckets holding both labels (mixing ratio
{C['mixing_ratio']:.2f}), and {C['pool_orphans']} held-out conditions without neighbors. It was
the simplest key with large neighborhoods, and none of the alternatives had clearly better
mixing.</p>
{img(os.path.join(out, "fig5_label_mix_chosen.png"), "70%")}
{''.join(img(g) for g in galleries)}

<h2>6. Trade-offs and open questions</h2>
<p>The chosen condition says nothing about supports: generated parts match the load octant and
volume fraction but may place supports anywhere. If BC control is needed later, adding the
load-to-BC distance is the next step up (see the table).</p>
<p>Because positive is defined against all parts pooled, the label depends strongly on shape.
Defining positive relative to each shape's own deformation distribution would make the label
mean &ldquo;good for its size&rdquo; and remove the shape floor. That changes what the label
means, so it is a modelling decision rather than a tuning one.</p>
<p>Next: train on the chosen condition, then generate parts for held-out conditions with
positive and negative requests and compare surrogate-scored outcomes, each against the real
training parts that share its condition.</p>
</body></html>"""
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(page)
    print(f"Wrote {out}/report.html, report.json and 5 figures"
          + (f" plus {len(galleries)} galleries" if galleries else ""))


if __name__ == "__main__":
    main()
