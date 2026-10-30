"""
Class-conditional sampler for the label-only manufacturability generator (v1008).

Generates N parts of a chosen shape for a chosen manufacturability class
("positive", "negative", or "unconditional"), with no TO spec. Intended for
checkpoints trained on v1008_make_train_data_labelonly.sh data, whose
condition vector is [manuf_label, manuf_has_label] (cond_dim = 2).

The evaluation question this sampler serves: if we ask for 100 positive parts,
what fraction does the surrogate score positive, and likewise for negative.
So the outputs are organized for scoring, not for comparison to ground truth
(there is no ground-truth part for a class request).

Design notes
------------
1. Paired noise. For a given (shape, batch chunk) the initial noise is the same
   for every requested label. Member k of "positive" and member k of "negative"
   therefore start from the identical noise draw, so differences between them
   come from the label alone. This makes paired comparisons possible, but the
   class fractions are still estimated from independent parts across k.
2. "unconditional" writes the all-zero condition, which is exactly what the
   trainer used for dropped conditions. It estimates the model's own base
   rate, the baseline the positive/negative fractions should move away from.
   (With unconditional, guidance has no effect, since cond and uncond agree.)
3. The training data's own statistics are written next to the samples:
   per shape, how many parts, the fraction positive, and the mean mass of
   positive and negative parts. Comparing generated mass to these shows
   whether the model changes the class mainly by adding or removing material.
4. Every part is listed in parts_index.csv with empty surrogate columns, ready
   for the surrogate run to fill in.
5. Scalar-mode checkpoints are accepted too: positive/negative map to the
   midpoint percentile on each side of the split (as in the v0804 sampler),
   or --scalar_value sets an exact percentile rank.
6. Any condition slots other than the manufacturability ones are zero-filled,
   with a loud warning: this sampler is for label-only models.

Requires in the same directory: v0720_neural_diff_trainer_fixed.py and
v0723_masked_voxel_diff_trainer.py.
"""

import argparse
import csv
import functools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from v0720_neural_diff_trainer_fixed import (
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    plot_voxel_with_overlays_implicit,
)
from v0723_masked_voxel_diff_trainer import (
    MaskedScoreNet3D,
    heun_sampler_masked,
    pad_to_multiple,
    speckle_fraction,
    pairwise_iou,
)

MANUF_SLOTS = {"manuf_label", "manuf_value", "manuf_has_label"}
LABEL_CHOICES = ("positive", "negative", "unconditional")


def parse_args():
    p = argparse.ArgumentParser(description="Label-only class-conditional sampler (v1008)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hparams", type=str, default=None,
                   help="Default: <ckpt_dir>/../hparams.json")
    p.add_argument("--data_file", type=str, required=True,
                   help="The label-only training .npy (for meta, shapes and reference stats).")
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)

    p.add_argument("--labels", type=str, default="positive,negative",
                   help="Comma list from: positive, negative, unconditional.")
    p.add_argument("--shapes", type=str, default="all",
                   help="'all' (every shape in the training data) or e.g. '64x32x16,32x32x32'.")
    p.add_argument("--n_per_group", default=16, type=int,
                   help="Parts generated per (label, shape) group.")
    p.add_argument("--batch_size", default=16, type=int,
                   help="Members per batched solve (memory knob only).")
    p.add_argument("--scalar_value", type=float, default=None,
                   help="Scalar-mode checkpoints only: exact percentile rank for 'positive'.")

    p.add_argument("--n_steps", default=250, type=int)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    p.add_argument("--guidance_scale", default=0.0, type=float)

    p.add_argument("--n_png_per_group", default=4, type=int,
                   help="Plot only the first K members of each group (plots are slow).")
    p.add_argument("--iou_max_members", default=16, type=int,
                   help="Pairwise IoU uses at most this many members per group.")
    return vars(p.parse_args())


def shape_str(s):
    return f"{s[0]}x{s[1]}x{s[2]}"


def parse_shapes(spec, available):
    if spec.strip().lower() == "all":
        return list(available)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Bad shape {tok!r}; use e.g. 64x32x16")
        s = tuple(int(v) for v in parts)
        if s not in available:
            print(f"WARNING: shape {s} does not occur in the training data; "
                  f"the model has never seen it. Generating anyway.")
        out.append(s)
    return out


def load_reference_stats(data_file):
    """Per-shape counts, positive fraction and mean mass by class, from the
    training .npy (sample_info entries only)."""
    data = np.load(data_file, allow_pickle=True)
    stats = {}
    for entry in data:
        info = entry[4] if len(entry) > 4 else {}
        s = tuple(int(v) for v in np.asarray(info["part_shape"]).tolist())
        d = stats.setdefault(s, {"n": 0, "n_pos": 0, "n_neg": 0, "n_unlab": 0,
                                 "mass_pos": [], "mass_neg": []})
        d["n"] += 1
        lbl = info.get("manuf_label")
        m = float(info.get("mass_fraction", np.nan))
        if not info.get("manuf_has_label", False):
            d["n_unlab"] += 1
        elif lbl is not None and lbl > 0:
            d["n_pos"] += 1
            d["mass_pos"].append(m)
        else:
            d["n_neg"] += 1
            d["mass_neg"].append(m)
    summary = {}
    for s in sorted(stats):
        d = stats[s]
        nl = d["n_pos"] + d["n_neg"]
        summary[shape_str(s)] = {
            "n_parts": d["n"],
            "n_positive": d["n_pos"],
            "n_negative": d["n_neg"],
            "n_unlabeled": d["n_unlab"],
            "frac_positive": round(d["n_pos"] / nl, 4) if nl else None,
            "mean_mass_positive": round(float(np.mean(d["mass_pos"])), 4) if d["mass_pos"] else None,
            "mean_mass_negative": round(float(np.mean(d["mass_neg"])), 4) if d["mass_neg"] else None,
        }
    tot_pos = sum(v["n_positive"] for v in summary.values())
    tot_lab = tot_pos + sum(v["n_negative"] for v in summary.values())
    return sorted(stats), summary, (tot_pos / tot_lab if tot_lab else None)


def build_cond(label, cond_dim, cond_slices, manuf_mode, meta, scalar_value):
    cond = np.zeros(cond_dim, dtype=np.float32)
    if label == "unconditional":
        return cond

    def put(name, val):
        s, e = cond_slices[name]
        cond[s:e] = val

    put("manuf_has_label", 1.0)
    if manuf_mode == "binary":
        put("manuf_label", 1.0 if label == "positive" else -1.0)
    elif manuf_mode == "scalar":
        thr = float(meta["manuf_percentile_threshold"]) / 100.0
        if label == "positive":
            v = scalar_value if scalar_value is not None else thr / 2.0
        else:
            v = thr + (1.0 - thr) / 2.0
        put("manuf_value", v)
    else:
        raise ValueError(f"Dataset manufacturability_mode is {manuf_mode!r}; "
                         "this sampler needs binary or scalar.")
    return cond


def main():
    args = parse_args()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    labels = [l.strip() for l in args["labels"].split(",") if l.strip()]
    for l in labels:
        if l not in LABEL_CHOICES:
            raise ValueError(f"Unknown label {l!r}; choose from {LABEL_CHOICES}")

    ckpt_path = Path(args["ckpt"])
    hparams_path = Path(args["hparams"]) if args["hparams"] else ckpt_path.parent.parent / "hparams.json"
    with open(hparams_path) as f:
        hp = json.load(f)
    meta_path = args["meta_path"] or hp.get("meta_path") or args["data_file"][:-4] + "_meta.npz"
    meta = np.load(meta_path, allow_pickle=True)
    print(f"Loaded hparams from {hparams_path}")

    manuf_mode = str(meta["manufacturability_mode"]) if "manufacturability_mode" in meta else "omit"
    cond_slices = json.loads(str(meta["cond_slices_json"]))
    cond_dim = int(meta["cond_dim"])
    extra = sorted(set(cond_slices) - MANUF_SLOTS)
    if extra:
        print("=" * 72)
        print(f"WARNING: this dataset's condition has non-manufacturability slots {extra}.")
        print("This sampler zero-fills them. It is meant for LABEL-ONLY checkpoints;")
        print("results from a spec-conditioned checkpoint will not be meaningful.")
        print("=" * 72)
    if manuf_mode != "binary" and manuf_mode != "scalar":
        raise ValueError(f"manufacturability_mode {manuf_mode!r} has no label to condition on.")
    if args["guidance_scale"] > 0 and hp.get("cond_drop_prob", 0.0) <= 0:
        print("WARNING: guidance_scale > 0 but cond_drop_prob = 0 in training; CFG is not valid.")

    model = MaskedScoreNet3D(
        cond_dim=cond_dim, cond_embed_dim=hp["cond_embed_dim"], cond_ch=hp["cond_ch"],
        c1=hp["unet_ch1_dim"], c2=hp["unet_ch2_dim"], c3=hp["unet_ch3_dim"],
        ed=hp["t_embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}  (cond_dim={cond_dim}, mode={manuf_mode})")

    available, ref_stats, ref_frac_pos = load_reference_stats(args["data_file"])
    shapes = parse_shapes(args["shapes"], available)
    print(f"Training-data positive fraction (all shapes): {ref_frac_pos:.4f}")
    print(f"Labels: {labels}   shapes: {[shape_str(s) for s in shapes]}   "
          f"n_per_group: {args['n_per_group']}   guidance: {args['guidance_scale']}")

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    outdir = Path(args["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    n_group, B = args["n_per_group"], max(1, args["batch_size"])
    n_chunks = math.ceil(n_group / B)
    index_rows, group_summaries = [], []

    for si, shp in enumerate(shapes):
        Nx, Ny, Nz = shp
        Px, Py, Pz = pad_to_multiple(Nx), pad_to_multiple(Ny), pad_to_multiple(Nz)
        for label in labels:
            cond_vec = build_cond(label, cond_dim, cond_slices, manuf_mode, meta, args["scalar_value"])
            fields, seeds = [], []
            for ch in range(n_chunks):
                b = min(B, n_group - ch * B)
                # paired noise: seed depends on (shape, chunk), not on the label
                noise_seed = args["seed"] * 1000003 + si * 10007 + ch
                gen = torch.Generator(device=device)
                gen.manual_seed(noise_seed)
                mask = torch.zeros(b, 1, Px, Py, Pz)
                mask[:, :, :Nx, :Ny, :Nz] = 1.0
                cond = torch.from_numpy(cond_vec)[None].expand(b, -1)
                x = heun_sampler_masked(
                    model, mask, cond, mean_fn, std_fn, drift_fn,
                    n_steps=args["n_steps"], eps=args["sample_eps"], device=device,
                    guidance_scale=args["guidance_scale"], generator=gen,
                ).numpy()[:, 0, :Nx, :Ny, :Nz]
                fields.append(x)
                seeds += [noise_seed] * b
            fields = np.concatenate(fields, axis=0)
            bins = (fields > 0.0).astype(np.float32)
            masses = [float(v) for v in bins.reshape(len(bins), -1).mean(axis=1)]
            speckles = [speckle_fraction(b_) for b_ in bins]
            k = min(len(bins), args["iou_max_members"])
            iou = pairwise_iou([bins[i] for i in range(k)])

            gname = f"{label}_{shape_str(shp)}"
            gdir = outdir / gname
            gdir.mkdir(exist_ok=True)
            np.savez_compressed(
                gdir / "ensemble.npz",
                fields=fields, binaries=bins, cond=cond_vec,
                requested_label=label, shape=np.asarray(shp),
                noise_seeds=np.asarray(seeds), guidance_scale=args["guidance_scale"],
                n_steps=args["n_steps"], sample_eps=args["sample_eps"],
                manuf_mode=manuf_mode,
            )
            for m in range(min(args["n_png_per_group"], len(bins))):
                plot_voxel_with_overlays_implicit(
                    bins[m], None, None, None,
                    title=(f"{label} | {shape_str(shp)} | member {m} | mass {masses[m]:.3f}"
                           f" | speckle {speckles[m]:.3f} | w={args['guidance_scale']}"),
                    save_path=gdir / f"member{m:03d}.png",
                )
            for m in range(len(bins)):
                index_rows.append({
                    "part_id": f"{gname}_{m:04d}",
                    "requested_label": label,
                    "shape": shape_str(shp),
                    "group_dir": gname,
                    "member_in_npz": m,
                    "noise_seed": seeds[m],
                    "mass_fraction": round(masses[m], 5),
                    "speckle_fraction": round(speckles[m], 5),
                    "surrogate_deformation_p99": "",
                    "surrogate_label": "",
                })
            ref = ref_stats.get(shape_str(shp), {})
            gs = {
                "group": gname, "requested_label": label, "shape": shape_str(shp),
                "n_parts": len(bins),
                "mean_mass": round(float(np.mean(masses)), 4),
                "std_mass": round(float(np.std(masses)), 4),
                "mean_speckle": round(float(np.mean(speckles)), 5),
                "pairwise_iou": round(iou, 4) if iou == iou else None,
                "train_mean_mass_positive": ref.get("mean_mass_positive"),
                "train_mean_mass_negative": ref.get("mean_mass_negative"),
                "train_frac_positive": ref.get("frac_positive"),
            }
            group_summaries.append(gs)
            print(f"[{gname}] n={len(bins)}  mass {gs['mean_mass']:.3f}+/-{gs['std_mass']:.3f} "
                  f"(train pos {gs['train_mean_mass_positive']}, neg {gs['train_mean_mass_negative']})  "
                  f"speckle {gs['mean_speckle']:.4f}  IoU {gs['pairwise_iou']}")

    with open(outdir / "parts_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    with open(outdir / "training_data_reference.json", "w") as f:
        json.dump({"overall_frac_positive": ref_frac_pos, "per_shape": ref_stats,
                   "threshold_value": float(meta["manuf_threshold_value"]),
                   "threshold_percentile": float(meta["manuf_percentile_threshold"]),
                   "scalar_column": str(meta["manuf_scalar_column"])}, f, indent=2)
    with open(outdir / "run_summary.json", "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "sampler": "v1008_classcond_sampler.py",
            "ckpt": str(ckpt_path), "data_file": str(args["data_file"]),
            "args": args, "groups": group_summaries,
        }, f, indent=2, default=str)
    print(f"\nWrote {len(index_rows)} parts. Score them with the surrogate and fill "
          f"the surrogate_* columns of {outdir / 'parts_index.csv'}.")
    print(f"Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()
