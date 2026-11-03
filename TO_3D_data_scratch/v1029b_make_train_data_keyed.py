"""
v1029b_make_train_data_keyed.py

Same as v1029_make_train_data_binned.py (80/20 stratified train/pool split,
bucket index, CSVs, exact geometry kept in sample_info), but the condition
vector and the bucket key are built from a chosen KEY of coarse fields
instead of the full binned spec.

Why (v1029 diagnostics, 2026-10-29, real data, 4,450 train parts): the full
key (load cell, BC cell pattern, VF bin, direction) gave 4,103 buckets, 85%
singletons, 2.2% of parts in mixed-label buckets, 946 of 1,114 pool
conditions with no neighbor. Key ablation: shape+load+vf gave 224 buckets,
median 20, 0 singletons, 69.5% of parts in mixed-label buckets (shuffle
ratio 0.71), 0 orphan pool conditions. Default key is therefore "load+vf".

Shape is always in the bucket key (it reaches the model through the domain
mask) and never in the condition vector. See v1029b_keyed_cond_utils.py for
the fields: load, vf, bccen, ldist, bcpat, dir.

--exclude-shapes drops whole shapes before the split (e.g. a shape whose
labels are almost all one class).

Outputs in --outdir, as in v1029:
  <n>_keyed_<fields>_k<k>_vf<n>_manuf-binary_p<P>_{train,pool}.npy (+ _meta.npz)
  train_parts.csv, pool_parts.csv, bucket_index.json, split_summary.json
The _meta.npz adds key_fields_json; cond_slices_json describes the vector.
Works unchanged with v1029_bucket_diagnostics.py and
v1029_visualize_train_data_binned.py.
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
from v1029_binned_cond_utils import DIR_CLASSES, vf_quantile_edges, cell_name
from v1029b_keyed_cond_utils import (
    parse_key, keyed_layout, all_field_values, keyed_bucket_key, encode_keyed,
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
    p.add_argument("--shapes-filter", default="", help="keep only these shapes")
    p.add_argument("--exclude-shapes", default="", help="drop these shapes, e.g. 120x20x20")
    p.add_argument("--key", default="load+vf",
                   help="fields joined by '+': load, vf, bccen, ldist, bcpat, dir (shape is implicit)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    k, n_vf = args.k_bins, args.n_vf_bins
    fields = parse_key(args.key)
    print(f"Key: shape + {' + '.join(fields)}")

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
    ex = parse_shapes_filter(args.exclude_shapes) or set()
    universe = sorted(i for i in labeled if 0 <= i < len(all_shapes)
                      and (sf is None or all_shapes[i] in sf) and all_shapes[i] not in ex)
    if ex:
        print(f"Excluding shapes: {sorted(ex)}")
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
    slices, cond_dim = keyed_layout(fields, k, n_vf)

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
            vals = all_field_values(shp, bc_pts, load_pt, load_vec, vf, k, vf_edges)
            cond = encode_keyed(fields, vals, lab, k, n_vf)
            dec = {"load_cell": vals["load"], "bc_cells": vals["bcpat"], "vf_bin": vals["vf"],
                   "load_dir_class": vals["dir"],
                   "bucket_key": keyed_bucket_key(shp, fields, vals)}
            if "dir" in fields and dec["load_dir_class"] < 0:
                n_zero_dir += 1
            mass = float((vox > 0).mean())
            cond_str = (f"cond_keyed_k{k}_{dec['bucket_key']}_vf_{vf:.6f}"
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
                "bc_centroid_cell": vals["bccen"],
                "load_bc_distance": vals["ldist"],
                "key_fields": list(fields),
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
                "bc_centroid_cell": cell_name(vals["bccen"], k),
                "load_bc_distance": vals["ldist"],
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
    tag = (f"keyed_{'-'.join(fields)}_k{k}_vf{n_vf}"
           f"_manuf-binary_p{args.manuf_percentile_threshold:g}")
    spec = {"key": "shape+" + "+".join(fields), "k_bins": k, "n_vf_bins": n_vf,
            "bc_dofs": "omit", "manufacturability": "binary"}
    common_meta = dict(
        conditioning_spec_json=json.dumps(spec),
        cond_slices_json=json.dumps(slices),
        cond_dim=int(cond_dim),
        k_bins=int(k), n_vf_bins=int(n_vf),
        vf_edges=np.asarray(vf_edges, dtype=np.float64),
        dir_classes_json=json.dumps(DIR_CLASSES),
        key_fields_json=json.dumps(fields),
        excluded_shapes_json=json.dumps(sorted(ex)),
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
        json.dump({"key": "shape+" + "+".join(fields), "k_bins": k, "n_vf_bins": n_vf, "vf_edges": list(map(float, vf_edges)),
                   "train_file": os.path.basename(paths["train"]), "buckets": buckets}, f)

    summary = {
        "train_file": paths["train"], "pool_file": paths["pool"],
        "n_universe": len(universe), "n_train": len(train_ids), "n_pool": len(pool_ids),
        "frac_positive_train": float(np.mean([labels[i] > 0 for i in train_ids])),
        "frac_positive_pool": float(np.mean([labels[i] > 0 for i in pool_ids])) if pool_ids else None,
        "manuf_threshold_value": float(thr), "cond_dim": int(cond_dim),
        "key": "shape+" + "+".join(fields), "excluded_shapes": [f"{a}x{b}x{c}" for a, b, c in sorted(ex)],
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
    print(f"Condition dim: {cond_dim}   (" + " + ".join(
        f"{b - a} {n.replace('_onehot', '').replace('_multihot', '')}"
        for n, (a, b) in slices.items() if not n.startswith("manuf")) + " + 2 manuf)")
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
