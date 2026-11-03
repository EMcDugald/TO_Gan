"""
v1029_visualize_train_data_binned.py

Sanity plots for the v1029 binned-condition data.

  parts/part_<split>_<pos>.png   one part, two views: voxels, BC cells as blue
                                 boxes, load cell as a red box, and the EXACT
                                 BC points (navy dots) and load point / vector
                                 (red triangle, arrow). Every dot should sit
                                 inside a blue box and the triangle inside the
                                 red box; if not, the binning is wrong.
  buckets/bucket_<rank>.png      up to --gallery_members parts that share one
                                 bucket key (a preview of the ground-truth
                                 neighborhoods the sampler will show).
  summary.png                    bucket sizes, label mix per bucket, VF with
                                 bin edges, load-cell / BC-cell / direction
                                 frequencies.

Usage:
  python v1029_visualize_train_data_binned.py --data_dir <outdir> [--split train]
      [--n_parts 8] [--n_buckets 6] [--gallery_members 4] [--seed 0]
      [--positions 3,17,42]
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v1029_binned_cond_utils import (
    DIR_CLASSES, cell_name, plot_part_binned, vf_bin_label,
)


def title_for(info, k, edges, thr_col="deformation"):
    s = info["part_shape"]
    lab = "positive" if info["manuf_label"] > 0 else "negative"
    d = DIR_CLASSES[info["load_dir_class"]] if info["load_dir_class"] >= 0 else "zero"
    return (f"{info['split']} pos {info['split_position']} | src {info['source_index']} | "
            f"{s[0]}x{s[1]}x{s[2]} | {lab} ({thr_col} {info['manuf_raw_value']:.3f})\n"
            f"load {cell_name(info['load_cell'], k)}  dir {d}  |  "
            f"BC cells {' '.join(cell_name(c, k) for c in info['bc_cells'])}  |  "
            f"{vf_bin_label(info['vf_bin'], edges)} (vf {info['volume_fraction']:.3f})  |  "
            f"mass {info['mass_fraction']:.3f}")


def plot_part(entry, k, edges, path):
    vox, info = entry[0], entry[4]
    fig = plt.figure(figsize=(12, 5.5))
    for j, (el, az) in enumerate([(25, 35), (25, 215)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        plot_part_binned(ax, vox, vox.shape, k, info["load_cell"], info["bc_cells"],
                         info["bc_points_exact"], info["load_point_exact"],
                         info["load_vec_exact"], elev=el, azim=az)
    fig.suptitle(title_for(info, k, edges), fontsize=9)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_bucket(entries, key, k, path):
    n = len(entries)
    fig = plt.figure(figsize=(4.2 * n, 4.6))
    for j, e in enumerate(entries):
        vox, info = e[0], e[4]
        ax = fig.add_subplot(1, n, j + 1, projection="3d")
        lab = "positive" if info["manuf_label"] > 0 else "negative"
        plot_part_binned(ax, vox, vox.shape, k, info["load_cell"], info["bc_cells"],
                         info["bc_points_exact"], info["load_point_exact"],
                         info["load_vec_exact"],
                         title=(f"#{info['split_position']} | {lab} | "
                                f"vf {info['volume_fraction']:.3f} | mass {info['mass_fraction']:.3f}"))
    fig.suptitle(key, fontsize=10)
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_summary(infos, k, edges, path):
    by_key = defaultdict(list)
    for inf in infos:
        by_key[inf["bucket_key"]].append(1 if inf["manuf_label"] > 0 else 0)
    sizes = np.array([len(v) for v in by_key.values()])
    fpos = np.array([np.mean(v) for v in by_key.values()])

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    ax.hist(sizes, bins=np.arange(1, sizes.max() + 2) - 0.5 if sizes.max() < 60 else 40,
            color="0.4")
    ax.set_xlabel("parts in bucket"); ax.set_ylabel("number of buckets")
    ax.set_title(f"{len(sizes)} buckets, median size {np.median(sizes):.0f}")

    ax = axs[0, 1]
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(sizes))
    ax.scatter(sizes + jitter, fpos, s=10, alpha=0.5)
    base = np.mean([1 if i["manuf_label"] > 0 else 0 for i in infos])
    ax.axhline(base, color="k", lw=0.8, ls="--", label=f"base rate {base:.2f}")
    ax.set_xscale("log"); ax.set_xlabel("parts in bucket (log)")
    ax.set_ylabel("fraction positive in bucket"); ax.legend(fontsize=8)
    ax.set_title("label mix per bucket (spread away from 0/1 = mixed)")

    ax = axs[0, 2]
    vf = np.array([i["volume_fraction"] for i in infos])
    ax.hist(vf, bins=40, color="0.4")
    for e in edges:
        ax.axvline(e, color="tab:red", lw=1)
    ax.set_xlabel("volume fraction"); ax.set_title("VF with bin edges (red)")

    n_cells = k ** 3
    names = [cell_name(c, k) for c in range(n_cells)]
    ax = axs[1, 0]
    lc = Counter(i["load_cell"] for i in infos)
    ax.bar(range(n_cells), [lc.get(c, 0) for c in range(n_cells)], color="tab:red")
    ax.set_xticks(range(n_cells)); ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_title("load cell frequency")

    ax = axs[1, 1]
    bcc = Counter(c for i in infos for c in i["bc_cells"])
    ax.bar(range(n_cells), [bcc.get(c, 0) for c in range(n_cells)], color="tab:blue")
    ax.set_xticks(range(n_cells)); ax.set_xticklabels(names, rotation=60, fontsize=7)
    nbc = np.array([len(i["bc_cells"]) for i in infos])
    ax.set_title(f"BC cell frequency (cells per part: median {np.median(nbc):.0f})")

    ax = axs[1, 2]
    dc = Counter(i["load_dir_class"] for i in infos)
    labels = DIR_CLASSES + (["zero"] if -1 in dc else [])
    vals = [dc.get(c, 0) for c in range(6)] + ([dc[-1]] if -1 in dc else [])
    ax.bar(range(len(vals)), vals, color="0.4")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
    ax.set_title("load direction class frequency")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Sanity plots for v1029 binned data.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=["train", "pool"], default="train")
    ap.add_argument("--outdir", default="", help="default: <data_dir>/viz_<split>")
    ap.add_argument("--n_parts", type=int, default=8)
    ap.add_argument("--positions", default="", help="comma-separated split positions (overrides n_parts)")
    ap.add_argument("--n_buckets", type=int, default=6)
    ap.add_argument("--gallery_members", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, f"*_{args.split}.npy")))
    if len(paths) != 1:
        raise SystemExit(f"Expected one *_{args.split}.npy in {args.data_dir}, found {paths}")
    data = np.load(paths[0], allow_pickle=True)
    meta = np.load(paths[0][:-4] + "_meta.npz", allow_pickle=True)
    k, edges = int(meta["k_bins"]), np.asarray(meta["vf_edges"])
    infos = [d[4] for d in data]
    outdir = args.outdir or os.path.join(args.data_dir, f"viz_{args.split}")
    os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "buckets"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    plot_summary(infos, k, edges, os.path.join(outdir, "summary.png"))
    print(f"Wrote {os.path.join(outdir, 'summary.png')}")

    if args.positions.strip():
        pos = [int(t) for t in args.positions.split(",") if t.strip()]
    else:
        pos = sorted(rng.choice(len(data), size=min(args.n_parts, len(data)), replace=False).tolist())
    for p in pos:
        plot_part(data[p], k, edges, os.path.join(outdir, "parts", f"part_{args.split}_{p:05d}.png"))
    print(f"Wrote {len(pos)} part plots to {os.path.join(outdir, 'parts')}")

    by_key = defaultdict(list)
    for i, inf in enumerate(infos):
        by_key[inf["bucket_key"]].append(i)
    eligible = [kk for kk, v in by_key.items() if len(v) >= 2]
    eligible.sort(key=lambda kk: -len(by_key[kk]))
    # half from the largest buckets, half at random among the rest
    n_top = min(len(eligible), (args.n_buckets + 1) // 2)
    chosen = eligible[:n_top]
    rest = eligible[n_top:]
    if rest:
        chosen += [rest[j] for j in rng.choice(len(rest), size=min(len(rest), args.n_buckets - n_top),
                                               replace=False)]
    gallery = []
    for r, kk in enumerate(chosen):
        mem = by_key[kk]
        pick = sorted(rng.choice(mem, size=min(len(mem), args.gallery_members), replace=False).tolist())
        plot_bucket([data[i] for i in pick], f"{kk}   ({len(mem)} parts in bucket)", k,
                    os.path.join(outdir, "buckets", f"bucket_{r:02d}.png"))
        gallery.append({"rank": r, "bucket_key": kk, "n_parts": len(mem), "shown_positions": pick})
    with open(os.path.join(outdir, "buckets", "gallery_index.json"), "w") as f:
        json.dump(gallery, f, indent=2)
    print(f"Wrote {len(chosen)} bucket galleries to {os.path.join(outdir, 'buckets')}")


if __name__ == "__main__":
    main()
