"""
v1029_make_train_data_binned.py

Builds the "binned TO spec" dataset for the v1029 manufacturability model and
splits it into a TRAIN file (default 80%) and a POOL file (default 20%) of
held-out conditions for sampling.

Why: with exact BC/load coordinates every TO spec belongs to exactly one part,
so the manufacturability label was redundant and the v0804 models ignored it
(positive and negative requests both came out ~80% positive). Binning makes
many parts share one condition; within a shared condition only the label
separates the parts, so the model has a reason to use it.

Condition vector (see v1029_binned_cond_utils.py): load cell one-hot (k^3),
BC cells multi-hot (k^3), VF bin one-hot (n_vf_bins), load-direction class
one-hot (6), [manuf_label +/-1, manuf_has_label]. BC DOFs omitted. Shape
reaches the model only through the domain mask, as before.

Universe: the parts with a finite deformation value in the CSV (~5,564),
optionally shape-filtered. Positive = value <= the given percentile of ALL
labeled parts (same threshold rule as v0804/v1008; p80 -> 1.10439).

Split: stratified by (shape, label) with --split-seed, so the pool has the
training mix. VF bin edges are quantiles of the TRAIN parts' VFs only.

Outputs in --outdir:
  <n>_binned_k<k>_vf<n>_manuf-binary_p<P>_train.npy  (+ _meta.npz)
  <n>_binned_k<k>_vf<n>_manuf-binary_p<P>_pool.npy   (+ _meta.npz)
  train_parts.csv, pool_parts.csv   one row per part
  bucket_index.json                 bucket key -> train positions / source
                                    indices / n_pos / n_neg
  split_summary.json                counts, threshold, bucket statistics

Each .npy entry keeps the format the trainer expects:
  (voxel int8 (Nx,Ny,Nz), cond_vec float32, 0, cond_str, sample_info)
sample_info also carries the bucket key, the decoded cells, and the EXACT BC
points / load point / load vector / VF (never fed to the model; used by the
plots and reports).

The existing data-gen files are imported read-only, never edited.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from v0804_cond_data_utils_neural_diff import (
    parse_bc_points, parse_load_point, parse_load_vector,
    load_manufacturability_scalar, compute_manufacturability_stats,
)
from v1029_binned_cond_utils import (
    DIR_CLASSES, cond_layout, encode_binned, vf_quantile_edges, cell_name,
)


def parse_shapes_filter(s):
    if not s.strip():
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            a = tok.lower().split("x")
            if len(a) != 3:
                raise ValueError(f"Invalid shape filter entry: {tok!r}")
            out.append(tuple(int(v) for v in a))
    return set(out)


def stratified_split(items, strata, pool_frac, seed):
    """items: list of ids; strata: parallel list of hashable keys."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for it, st in zip(items, strata):
        groups[st].append(it)
    train, pool = [], []
    for st in sorted(groups, key=str):
        g = np.array(sorted(groups[st]))
        g = g[rng.permutation(len(g))]
        n_pool = int(round(pool_frac * len(g)))
        if len(g) >= 2:
            n_pool = min(max(n_pool, 1 if pool_frac > 0 else 0), len(g) - 1)
        else:
            n_pool = 0
        pool += g[:n_pool].tolist()
        train += g[n_pool:].tolist()
    return sorted(train), sorted(pool)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--outdir", required=True)
    p.add_argument("--manufacturability-csv", required=True)
    p.add_argument("--manuf-scalar-column", default="deformation_p99")
    p.add_argument("--manuf-percentile-threshold", type=float, default=80.0)
    p.add_argument("--k-bins", type=int, default=2, help="cells per axis (2 = octants)")
    p.add_argument("--n-vf-bins", type=int, default=4)
    p.add_argument("--pool-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--shapes-filter", default="")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    k, n_vf = args.k_bins, args.n_vf_bins

    topo = np.load(os.path.join(args.data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(args.data_root, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(args.data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(args.data_root, "loads.npy"), allow_pickle=True)
    vfs = np.load(os.path.join(args.data_root, "vfs.npy"), allow_pickle=True).reshape(-1)
    all_shapes = [tuple(int(v) for v in np.asarray(s).tolist()) for s in shapes]
    print(f"Raw arrays: {len(all_shapes)} parts")

    labeled = load_manufacturability_scalar(args.manufacturability_csv, args.manuf_scalar_column)
    stats = compute_manufacturability_stats(labeled, args.manuf_percentile_threshold)
    thr = stats["threshold_value"]
    print(f"Labeled parts in CSV: {stats['n_labeled_total']}  "
          f"p{args.manuf_percentile_threshold:g} threshold = {thr:.6g} (<= threshold -> positive)")

    sf = parse_shapes_filter(args.shapes_filter)
    universe = sorted(i for i in labeled if 0 <= i < len(all_shapes)
                      and (sf is None or all_shapes[i] in sf))
    n_out_of_range = sum(1 for i in labeled if not (0 <= i < len(all_shapes)))
    if n_out_of_range:
        print(f"WARNING: {n_out_of_range} CSV part indices are outside the raw arrays; skipped.")
    labels = {i: (1.0 if labeled[i] <= thr else -1.0) for i in universe}
    print(f"Universe: {len(universe)} labeled parts "
          f"({sum(1 for v in labels.values() if v > 0)} positive)")

    train_ids, pool_ids = stratified_split(
        universe, [(all_shapes[i], labels[i]) for i in universe], args.pool_frac, args.split_seed)
    print(f"Split (seed {args.split_seed}, pool_frac {args.pool_frac}): "
          f"{len(train_ids)} train / {len(pool_ids)} pool")

    vf_edges = vf_quantile_edges(vfs[train_ids], n_vf)
    print("VF bin edges (train quantiles): " + ", ".join(f"{e:.4f}" for e in vf_edges))
    slices, cond_dim = cond_layout(k, n_vf)

    def build(ids, split):
        entries, rows = [], []
        n_zero_dir = 0
        for pos, idx in enumerate(ids):
            shp = all_shapes[idx]
            vox = np.asarray(topo[idx]).astype(np.int8).reshape(shp)
            bc_pts = parse_bc_points(bcs[idx])
            load_pt = parse_load_point(loads[idx])
            load_vec = parse_load_vector(loads[idx])
            vf = float(vfs[idx])
            lab = labels[idx]
            raw = float(labeled[idx])
            cond, dec = encode_binned(shp, bc_pts, load_pt, load_vec, vf, lab, k, vf_edges)
            if dec["load_dir_class"] < 0:
                n_zero_dir += 1
            mass = float((vox > 0).mean())
            cond_str = (f"cond_binned_k{k}_{dec['bucket_key']}_vf_{vf:.6f}"
                        f"_manuf_{args.manuf_scalar_column}_{raw:.6f}_lbl_{int(lab)}")
            info = {
                "source_index": int(idx),
                "part_shape": np.asarray(shp, dtype=np.int32),
                "bc_count": int(len(bc_pts)),
                "mass_fraction": mass,
                "volume_fraction": vf,
                "manuf_has_label": True,
                "manuf_raw_value": raw,
                "manuf_percentile_rank": float(stats["percentile_rank_fn"](raw)),
                "manuf_label": lab,
                "split": split,
                "split_position": int(pos),
                "bucket_key": dec["bucket_key"],
                "load_cell": dec["load_cell"],
                "bc_cells": dec["bc_cells"],
                "vf_bin": dec["vf_bin"],
                "load_dir_class": dec["load_dir_class"],
                "bc_points_exact": bc_pts.astype(np.float32),
                "load_point_exact": load_pt.astype(np.float32),
                "load_vec_exact": load_vec.astype(np.float32),
            }
            entries.append((vox, cond, int(0), cond_str, info))
            rows.append({
                "position": pos, "source_index": idx,
                "shape": f"{shp[0]}x{shp[1]}x{shp[2]}",
                "bucket_key": dec["bucket_key"],
                "load_cell": cell_name(dec["load_cell"], k),
                "bc_cells": " ".join(cell_name(c, k) for c in dec["bc_cells"]),
                "vf": f"{vf:.6f}", "vf_bin": dec["vf_bin"],
                "load_dir": DIR_CLASSES[dec["load_dir_class"]] if dec["load_dir_class"] >= 0 else "zero",
                "manuf_label": int(lab), args.manuf_scalar_column: f"{raw:.6f}",
                "mass_fraction": f"{mass:.5f}",
            })
        if n_zero_dir:
            print(f"WARNING: {n_zero_dir} {split} parts have a zero load vector "
                  f"(direction one-hot left all zeros, key 'Dzero').")
        return entries, rows

    train_entries, train_rows = build(train_ids, "train")
    pool_entries, pool_rows = build(pool_ids, "pool")

    # ---- bucket index (train only; neighborhoods ignore the label) ----
    buckets = defaultdict(lambda: {"train_positions": [], "source_indices": [], "n_pos": 0, "n_neg": 0})
    for r in train_rows:
        b = buckets[r["bucket_key"]]
        b["train_positions"].append(r["position"])
        b["source_indices"].append(r["source_index"])
        b["n_pos" if r["manuf_label"] > 0 else "n_neg"] += 1
    buckets = dict(sorted(buckets.items()))
    for r in pool_rows:
        b = buckets.get(r["bucket_key"])
        r["n_train_in_bucket"] = len(b["train_positions"]) if b else 0
    n_pool_orphan = sum(1 for r in pool_rows if r["n_train_in_bucket"] == 0)

    sizes = np.array([len(v["train_positions"]) for v in buckets.values()])
    part_sizes = np.array([len(buckets[r["bucket_key"]]["train_positions"]) for r in train_rows])
    mixed = np.array([buckets[r["bucket_key"]]["n_pos"] > 0 and buckets[r["bucket_key"]]["n_neg"] > 0
                      for r in train_rows])

    # ---- save ----
    tag = f"binned_k{k}_vf{n_vf}_manuf-binary_p{args.manuf_percentile_threshold:g}"
    spec = {"bc_locations": f"cells_k{k}", "bc_dofs": "omit", "load_location": f"cell_k{k}",
            "load_direction": "class6", "volume_fraction": f"bins{n_vf}",
            "manufacturability": "binary"}
    common_meta = dict(
        conditioning_spec_json=json.dumps(spec),
        cond_slices_json=json.dumps(slices),
        cond_dim=int(cond_dim),
        k_bins=int(k), n_vf_bins=int(n_vf),
        vf_edges=np.asarray(vf_edges, dtype=np.float64),
        dir_classes_json=json.dumps(DIR_CLASSES),
        vf_mode="binned", manufacturability_mode="binary",
        manuf_scalar_column=args.manuf_scalar_column,
        manuf_percentile_threshold=float(args.manuf_percentile_threshold),
        manuf_threshold_value=float(thr),
        manuf_n_labeled_total=int(stats["n_labeled_total"]),
        split_seed=int(args.split_seed), pool_frac=float(args.pool_frac),
        unique_shapes=np.asarray(sorted(set(all_shapes[i] for i in universe)), dtype=int),
    )
    paths = {}
    for split, entries in [("train", train_entries), ("pool", pool_entries)]:
        fname = f"{len(entries)}_{tag}_{split}.npy"
        path = os.path.join(args.outdir, fname)
        np.save(path, np.array(entries, dtype=object), allow_pickle=True)
        np.savez(path[:-4] + "_meta.npz", split=split, n_parts=len(entries), **common_meta)
        paths[split] = path
        print(f"Saved {split}: {path}")

    for split, rows in [("train", train_rows), ("pool", pool_rows)]:
        with open(os.path.join(args.outdir, f"{split}_parts.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(args.outdir, "bucket_index.json"), "w") as f:
        json.dump({"k_bins": k, "n_vf_bins": n_vf, "vf_edges": list(map(float, vf_edges)),
                   "train_file": os.path.basename(paths["train"]), "buckets": buckets}, f)

    summary = {
        "train_file": paths["train"], "pool_file": paths["pool"],
        "n_universe": len(universe), "n_train": len(train_ids), "n_pool": len(pool_ids),
        "frac_positive_train": float(np.mean([labels[i] > 0 for i in train_ids])),
        "frac_positive_pool": float(np.mean([labels[i] > 0 for i in pool_ids])) if pool_ids else None,
        "manuf_threshold_value": float(thr), "cond_dim": int(cond_dim),
        "k_bins": k, "n_vf_bins": n_vf, "vf_edges": list(map(float, vf_edges)),
        "n_train_buckets": int(len(sizes)),
        "bucket_size_median": float(np.median(sizes)), "bucket_size_max": int(sizes.max()),
        "frac_train_parts_in_singleton_buckets": float(np.mean(part_sizes == 1)),
        "frac_train_parts_in_buckets_ge5": float(np.mean(part_sizes >= 5)),
        "frac_train_parts_in_mixed_label_buckets": float(mixed.mean()),
        "n_pool_parts_with_empty_train_bucket": int(n_pool_orphan),
    }
    with open(os.path.join(args.outdir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(f"Condition dim: {cond_dim}   ({k**3} load cells + {k**3} BC cells + "
          f"{n_vf} VF bins + 6 directions + 2 manuf)")
    print(f"Train buckets: {len(sizes)}   median size {np.median(sizes):.0f}   max {sizes.max()}")
    print(f"Train parts in singleton buckets: {100*np.mean(part_sizes == 1):.1f}%   "
          f"in buckets of >=5: {100*np.mean(part_sizes >= 5):.1f}%")
    print(f"Train parts in buckets holding BOTH labels: {100*mixed.mean():.1f}%")
    if n_pool_orphan:
        print(f"WARNING: {n_pool_orphan} of {len(pool_rows)} POOL PARTS HAVE NO TRAINING PART "
              f"IN THEIR BUCKET (no ground-truth neighbors for those conditions).")
    print("Run v1029_bucket_diagnostics.py on this folder next.")


if __name__ == "__main__":
    main()
