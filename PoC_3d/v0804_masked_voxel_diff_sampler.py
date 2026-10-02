# """
# Standalone ensemble sampler for the Route C masked voxel diffusion model,
# with manufacturability-slot override support (v0804).

# Loads a checkpoint + its hparams.json (auto-inferred from the ckpt dir),
# picks conditions from the dataset (specific indices or seeded random draws),
# and generates an ensemble per condition with the batched Heun sampler.

#   --guidance_scale w   classifier-free guidance (valid because the trainer
#                        drops the condition with prob cond_drop_prob):
#                          x0hat = (1+w)*x0hat_cond - w*x0hat_uncond
#                        w=0 -> plain conditional sampling (max diversity).

#   --manuf_override     for datasets with manufacturability conditioning
#                        (meta.npz manufacturability_mode != "omit"), forces
#                        every sampled condition's manufacturability slot(s)
#                        to a requested target, overriding whatever that
#                        dataset entry actually recorded:
#                          none      -- leave as recorded (default)
#                          positive  -- force positive/manufacturable
#                          negative  -- force negative/non-manufacturable
#                          empty     -- force has_label=0 (sample as unlabeled)
#                          value     -- scalar-mode only, exact target via
#                                       --manuf_override_percentile or
#                                       --manuf_override_raw_value (+
#                                       --manufacturability_csv)
#                        BC/load/VF are still drawn from the chosen dataset
#                        index -- only the manufacturability field(s) change.

# Outputs per condition: GT png, member pngs (titled with manuf status when
# applicable), ensemble.npz (continuous fields, binaries, cond info, applied
# manuf override), and printed mass / speckle / pairwise-IoU / manuf stats.

# Requires v0720_neural_diff_trainer_fixed.py, v0723_masked_voxel_diff_trainer.py,
# and (only if using --manuf_override_raw_value) v0804_cond_data_utils_neural_diff.py
# in the same directory.
# """

# import argparse
# import functools
# import json
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import torch

# from v0720_neural_diff_trainer_fixed import (
#     NeuralFieldDataset3D,
#     marginal_prob_mean,
#     marginal_prob_std,
#     drift_coeff,
#     plot_voxel_with_overlays_implicit,
#     decode_condition_overlay,
# )
# from v0723_masked_voxel_diff_trainer import (
#     MaskedScoreNet3D,
#     heun_sampler_masked,
#     pad_to_multiple,
#     speckle_fraction,
#     pairwise_iou,
# )


# def decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode):
#     """Reads the manufacturability slot(s) straight out of a cond_vec, for
#     printing/manifest purposes -- independent of whatever override logic
#     was used to write them."""
#     if manuf_mode == "omit":
#         return {"has_label": None, "value": None, "label": None}

#     def get(name):
#         if name not in cond_slices:
#             return None
#         s, e = cond_slices[name]
#         return float(cond_vec[s:e][0])

#     hl = get("manuf_has_label")
#     has_label = bool(round(hl)) if hl is not None else None
#     value = get("manuf_value") if manuf_mode == "scalar" else None
#     label = get("manuf_label") if manuf_mode == "binary" else None
#     return {"has_label": has_label, "value": value, "label": label}


# def compute_manuf_override_targets(manuf_mode, meta, args):
#     """
#     Resolves --manuf_override into the concrete (value, label) to write,
#     validating mode-specific combinations up front so bad CLI combos fail
#     fast with a clear message rather than silently writing nothing useful.
#     Returns (target_value_or_None, target_label_or_None) -- exactly one is
#     populated depending on manuf_mode, for override modes that need a value.
#     """
#     override = args["manuf_override"]
#     if override == "none" or manuf_mode == "omit":
#         return None, None

#     threshold_pctl = float(meta["manuf_percentile_threshold"]) / 100.0  # e.g. 0.80

#     if override == "empty":
#         return None, None  # apply_manuf_override handles "empty" separately

#     if override in ("positive", "negative"):
#         if manuf_mode == "binary":
#             return None, (1.0 if override == "positive" else -1.0)
#         elif manuf_mode == "scalar":
#             # No discrete label field in scalar mode -- use a representative
#             # percentile on the correct side of the recorded split.
#             if override == "positive":
#                 repr_pctl = threshold_pctl / 2.0
#             else:
#                 repr_pctl = threshold_pctl + (1.0 - threshold_pctl) / 2.0
#             print(f"  (scalar-mode {override} override -> representative "
#                   f"percentile {repr_pctl:.4f}, split threshold is "
#                   f"{threshold_pctl:.4f})")
#             return repr_pctl, None

#     if override == "value":
#         if manuf_mode != "scalar":
#             raise ValueError("--manuf_override value is only valid for "
#                              "scalar-mode datasets; this dataset's "
#                              "manufacturability_mode is "
#                              f"{manuf_mode!r}. Use positive/negative/empty "
#                              "instead.")
#         if args["manuf_override_percentile"] is not None:
#             return float(args["manuf_override_percentile"]), None
#         if args["manuf_override_raw_value"] is not None:
#             if not args["manufacturability_csv"]:
#                 raise ValueError("--manuf_override_raw_value requires "
#                                  "--manufacturability_csv (to recompute the "
#                                  "percentile rank against the same labeled "
#                                  "population used at data-gen time).")
#             import sys
#             sys.path.insert(0, str(Path(__file__).resolve().parent))
#             from v0804_cond_data_utils_neural_diff import (
#                 load_manufacturability_scalar, compute_manufacturability_stats,
#             )
#             scalar_col = str(meta["manuf_scalar_column"])
#             labeled_map = load_manufacturability_scalar(
#                 args["manufacturability_csv"], scalar_col)
#             stats = compute_manufacturability_stats(
#                 labeled_map, float(meta["manuf_percentile_threshold"]))
#             pctl = stats["percentile_rank_fn"](float(args["manuf_override_raw_value"]))
#             print(f"  (raw value {args['manuf_override_raw_value']} -> "
#                   f"percentile rank {pctl:.4f} against "
#                   f"{stats['n_labeled_total']} labeled parts)")
#             return pctl, None
#         raise ValueError("--manuf_override value requires either "
#                          "--manuf_override_percentile or "
#                          "--manuf_override_raw_value (+ "
#                          "--manufacturability_csv).")

#     raise ValueError(f"Unhandled --manuf_override: {override}")


# def apply_manuf_override(cond_vec, cond_slices, manuf_mode, override,
#                          target_value=None, target_label=None):
#     if manuf_mode == "omit" or override == "none":
#         return cond_vec

#     cond_vec = cond_vec.copy()

#     def set_slice(name, val):
#         s, e = cond_slices[name]
#         cond_vec[s:e] = val

#     if override == "empty":
#         if manuf_mode == "scalar":
#             set_slice("manuf_value", 0.0)
#         elif manuf_mode == "binary":
#             set_slice("manuf_label", 0.0)
#         set_slice("manuf_has_label", 0.0)
#         return cond_vec

#     set_slice("manuf_has_label", 1.0)
#     if manuf_mode == "scalar":
#         set_slice("manuf_value", float(target_value))
#     elif manuf_mode == "binary":
#         set_slice("manuf_label", float(target_label))
#     return cond_vec


# def parse_args():
#     p = argparse.ArgumentParser(
#         description="Sample ensembles from a trained masked voxel diffusion model")
#     p.add_argument("--ckpt", type=str, required=True,
#                    help="Path to ckpt_best.pth (or any ckpt_*.pth)")
#     p.add_argument("--hparams", type=str, default=None,
#                    help="Path to hparams.json; default: <ckpt_dir>/../hparams.json")
#     p.add_argument("--data_file", type=str, required=True)
#     p.add_argument("--meta_path", type=str, default=None)
#     p.add_argument("--outdir", type=str, required=True)
#     p.add_argument("--device", default="cuda", type=str)
#     p.add_argument("--seed", default=0, type=int)

#     # which conditions
#     p.add_argument("--indices", type=str, default="",
#                    help="Comma-separated dataset indices, e.g. '3,17,102'. "
#                         "Empty -> draw --n_random random conditions.")
#     p.add_argument("--n_random", default=4, type=int)

#     # sampling
#     p.add_argument("--ensemble_size", default=8, type=int)
#     p.add_argument("--n_steps", default=250, type=int)
#     p.add_argument("--sample_eps", default=1e-3, type=float)
#     p.add_argument("--guidance_scale", default=0.0, type=float)

#     # manufacturability override -- if the dataset's meta.npz records
#     # manufacturability_mode != "omit", this lets you request a specific
#     # manufacturability target for every sampled condition, overriding
#     # whatever that dataset entry actually recorded (which may itself be
#     # labeled, unlabeled, positive, or negative).
#     p.add_argument("--manuf_override", type=str, default="none",
#                    choices=["none", "positive", "negative", "empty", "value"],
#                    help="none: leave the drawn condition's manufacturability "
#                         "fields exactly as recorded in the dataset. "
#                         "positive/negative: force a positive/negative label "
#                         "(binary-mode datasets: writes +1/-1; scalar-mode "
#                         "datasets: writes a representative percentile below/"
#                         "above the recorded split threshold). "
#                         "empty: force has_label=0, i.e. sample as if this "
#                         "condition had no manufacturability information. "
#                         "value: scalar-mode only -- write an exact target "
#                         "via --manuf_override_percentile or "
#                         "--manuf_override_raw_value.")
#     p.add_argument("--manuf_override_percentile", type=float, default=None,
#                    help="Scalar-mode + --manuf_override value: percentile "
#                         "rank in (0,1] to write directly into the manuf_value "
#                         "slot (same units the model was trained on).")
#     p.add_argument("--manuf_override_raw_value", type=float, default=None,
#                    help="Scalar-mode + --manuf_override value: a raw scalar "
#                         "value (e.g. a deformation_p99 number) to convert to "
#                         "a percentile rank via --manufacturability_csv before "
#                         "writing into the manuf_value slot.")
#     p.add_argument("--manufacturability_csv", type=str, default=None,
#                    help="Required only if --manuf_override_raw_value is used: "
#                         "path to the same summary CSV used at data-gen time, "
#                         "so the raw value can be converted to a percentile "
#                         "rank against the identical labeled population.")
#     return vars(p.parse_args())


# def main():
#     args = parse_args()
#     device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(args["seed"])
#     np.random.seed(args["seed"])

#     ckpt_path = Path(args["ckpt"])
#     hparams_path = (Path(args["hparams"]) if args["hparams"] is not None
#                     else ckpt_path.parent.parent / "hparams.json")
#     with open(hparams_path) as f:
#         hp = json.load(f)
#     print(f"Loaded hparams from {hparams_path}")

#     meta_path = args["meta_path"]
#     if meta_path is None:
#         meta_path = (hp.get("meta_path")
#                      or args["data_file"][:-4] + "_meta.npz")
#     meta = np.load(meta_path, allow_pickle=True)

#     dataset = NeuralFieldDataset3D(args["data_file"])

#     model = MaskedScoreNet3D(
#         cond_dim=dataset.cond_dim,
#         cond_embed_dim=hp["cond_embed_dim"],
#         cond_ch=hp["cond_ch"],
#         c1=hp["unet_ch1_dim"],
#         c2=hp["unet_ch2_dim"],
#         c3=hp["unet_ch3_dim"],
#         ed=hp["t_embed_dim"],
#     ).to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()
#     print(f"Loaded checkpoint {ckpt_path}")

#     if args["guidance_scale"] > 0.0 and hp.get("cond_drop_prob", 0.0) <= 0.0:
#         print("WARNING: guidance_scale > 0 but the model was trained with "
#               "cond_drop_prob = 0; CFG is not valid for this checkpoint.")

#     manuf_mode = str(meta["manufacturability_mode"]) if "manufacturability_mode" in meta else "omit"
#     cond_slices = json.loads(str(meta["cond_slices_json"])) if "cond_slices_json" in meta else {}
#     if manuf_mode != "omit":
#         print(f"Dataset manufacturability_mode = {manuf_mode} "
#               f"(threshold @ p{float(meta['manuf_percentile_threshold']):.0f} "
#               f"= {float(meta['manuf_threshold_value']):.6g})")
#     elif args["manuf_override"] != "none":
#         print("WARNING: --manuf_override was set but this dataset's "
#               "manufacturability_mode is 'omit' -- override will be ignored.")

#     manuf_target_value, manuf_target_label = compute_manuf_override_targets(
#         manuf_mode, meta, args)
#     if args["manuf_override"] != "none" and manuf_mode != "omit":
#         print(f"Applying manuf_override={args['manuf_override']!r} to every "
#               f"sampled condition")

#     if args["indices"].strip():
#         chosen = [int(s) for s in args["indices"].split(",") if s.strip()]
#     else:
#         rng = np.random.default_rng(args["seed"])
#         chosen = rng.choice(len(dataset), size=args["n_random"],
#                             replace=False).tolist()
#     print(f"Sampling conditions (dataset indices): {chosen}")

#     mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
#     std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
#     drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

#     outdir = Path(args["outdir"])
#     outdir.mkdir(parents=True, exist_ok=True)
#     E = args["ensemble_size"]

#     for ci, idx in enumerate(chosen):
#         idx = int(idx)
#         vox = dataset.voxels[idx]
#         Nx, Ny, Nz = vox.shape
#         Px, Py, Pz = (pad_to_multiple(Nx), pad_to_multiple(Ny),
#                       pad_to_multiple(Nz))
#         mask = torch.zeros(E, 1, Px, Py, Pz)
#         mask[:, :, :Nx, :Ny, :Nz] = 1.0
#         cond_vec = dataset.conds[idx]
#         original_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
#         cond_vec = apply_manuf_override(
#             cond_vec, cond_slices, manuf_mode, args["manuf_override"],
#             target_value=manuf_target_value, target_label=manuf_target_label,
#         )
#         applied_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
#         cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)

#         gen = torch.Generator(device=device)
#         gen.manual_seed(args["seed"] * 1000 + ci)
#         x_final = heun_sampler_masked(
#             model, mask, cond, mean_fn, std_fn, drift_fn,
#             n_steps=args["n_steps"], eps=args["sample_eps"], device=device,
#             guidance_scale=args["guidance_scale"], generator=gen,
#         ).numpy()[:, 0, :Nx, :Ny, :Nz]

#         bins = [(x_final[e] > 0.0).astype(np.float32) for e in range(E)]
#         masses = [float(b.mean()) for b in bins]
#         speckles = [speckle_fraction(b) for b in bins]
#         iou = pairwise_iou(bins)
#         gt_mass = float(vox.mean())

#         overlay = decode_condition_overlay(cond_vec, meta)
#         cdir = outdir / f"cond{ci:02d}_idx{idx}"
#         cdir.mkdir(exist_ok=True)

#         def _manuf_title():
#             if manuf_mode == "omit":
#                 return ""
#             if not applied_manuf["has_label"]:
#                 return " | manuf: UNLABELED"
#             if manuf_mode == "scalar":
#                 return f" | manuf: value={applied_manuf['value']:.4f}"
#             return f" | manuf: {'POSITIVE' if applied_manuf['label'] > 0 else 'NEGATIVE'}"

#         manuf_title_suffix = _manuf_title()

#         for e in range(E):
#             plot_voxel_with_overlays_implicit(
#                 bins[e], overlay["bc_points"], overlay["load_point"],
#                 overlay["load_dir"],
#                 title=(f"idx {idx} member {e} | ({Nx},{Ny},{Nz}) | "
#                        f"mass {masses[e]:.3f} | speckle {speckles[e]:.3f}"
#                        f"{manuf_title_suffix}"),
#                 save_path=cdir / f"member{e:02d}.png",
#             )
#         plot_voxel_with_overlays_implicit(
#             vox, overlay["bc_points"], overlay["load_point"],
#             overlay["load_dir"],
#             title=f"GT idx {idx} | ({Nx},{Ny},{Nz}) | mass {gt_mass:.3f}",
#             save_path=cdir / "gt.png",
#         )
#         np.savez_compressed(
#             cdir / "ensemble.npz",
#             fields=x_final, binaries=np.stack(bins),
#             cond=cond_vec, gt=vox, dataset_index=idx,
#             cond_str=str(dataset.cond_strs[idx]),
#             guidance_scale=args["guidance_scale"],
#             manuf_mode=manuf_mode,
#             manuf_override=args["manuf_override"],
#         )

#         # Human-readable provenance + decoded conditions. Coordinates are in
#         # the dataset convention: [0,1], normalized by max(Nx,Ny,Nz); multiply
#         # by (max_dim - 1) for voxel indices.
#         def _tolist(a):
#             return None if a is None else np.asarray(a, dtype=float).tolist()

#         sample_info = dataset.sample_infos[idx]
#         manifest = {
#             "provenance": {
#                 "dataset_index": idx,
#                 "data_file": str(args["data_file"]),
#                 "meta_path": str(meta_path),
#                 "ckpt": str(ckpt_path),
#                 "generated": datetime.now().isoformat(timespec="seconds"),
#             },
#             "part": {
#                 "shape": [Nx, Ny, Nz],
#                 "gt_mass_fraction": gt_mass,
#                 "cond_str": str(dataset.cond_strs[idx]),
#                 "sample_info": {k: (v.tolist() if isinstance(v, np.ndarray)
#                                     else v)
#                                 for k, v in dict(sample_info).items()}
#                                if isinstance(sample_info, dict) else None,
#             },
#             "conditions_decoded": {
#                 "coordinate_convention":
#                     "[0,1] normalized by max(Nx,Ny,Nz); "
#                     "voxel index ~= coord * (max_dim - 1)",
#                 "bc_points": _tolist(overlay["bc_points"]),
#                 "bc_dofs": _tolist(overlay.get("bc_dofs")),
#                 "load_point": _tolist(overlay["load_point"]),
#                 "load_dir": _tolist(overlay["load_dir"]),
#                 "conditioning_spec": overlay["conditioning_spec"],
#                 "cond_vector": cond_vec.astype(float).tolist(),
#             },
#             "manufacturability": {
#                 "mode": manuf_mode,
#                 "override_requested": args["manuf_override"],
#                 "as_originally_drawn": original_manuf,
#                 "as_applied_to_this_sample": applied_manuf,
#             },
#             "sampling": {
#                 "ensemble_size": E,
#                 "n_steps": args["n_steps"],
#                 "sample_eps": args["sample_eps"],
#                 "guidance_scale": args["guidance_scale"],
#                 "seed": args["seed"],
#                 "noise_seed": args["seed"] * 1000 + ci,
#             },
#             "results": {
#                 "member_mass_fractions": [round(m, 5) for m in masses],
#                 "member_speckle_fractions": [round(s, 5) for s in speckles],
#                 "pairwise_iou": round(iou, 5) if iou == iou else None,
#             },
#         }
#         with open(cdir / "manifest.json", "w") as jf:
#             json.dump(manifest, jf, indent=2, default=str)

#         print(f"[cond {ci}] idx {idx} shape ({Nx},{Ny},{Nz})  "
#               f"GT mass {gt_mass:.3f}")
#         print(f"          member masses  {[round(m,3) for m in masses]}")
#         print(f"          speckle fracs  {[round(s,3) for s in speckles]}  "
#               f"(GT ~ 0.00-0.02; target < 0.05)")
#         print(f"          pairwise IoU   {iou:.3f}  (target 0.5-0.8)")
#         if manuf_mode != "omit":
#             print(f"          manuf (drawn)   {original_manuf}")
#             print(f"          manuf (applied) {applied_manuf}")

#     print(f"Done. Outputs in {outdir}")


# if __name__ == "__main__":
#     main()


"""
Standalone ensemble sampler for the Route C masked voxel diffusion model,
with manufacturability-slot override support (v0804).

Loads a checkpoint + its hparams.json (auto-inferred from the ckpt dir),
picks conditions from the dataset (specific indices or seeded random draws),
and generates an ensemble per condition with the batched Heun sampler.

  --guidance_scale w   classifier-free guidance (valid because the trainer
                       drops the condition with prob cond_drop_prob):
                         x0hat = (1+w)*x0hat_cond - w*x0hat_uncond
                       w=0 -> plain conditional sampling (max diversity).

  --manuf_override     for datasets with manufacturability conditioning
                       (meta.npz manufacturability_mode != "omit"), forces
                       every sampled condition's manufacturability slot(s)
                       to a requested target, overriding whatever that
                       dataset entry actually recorded:
                         none      -- leave as recorded (default)
                         positive  -- force positive/manufacturable
                         negative  -- force negative/non-manufacturable
                         empty     -- force has_label=0 (sample as unlabeled)
                         value     -- scalar-mode only, exact target via
                                      --manuf_override_percentile or
                                      --manuf_override_raw_value (+
                                      --manufacturability_csv)
                       BC/load/VF are still drawn from the chosen dataset
                       index -- only the manufacturability field(s) change.

Outputs per condition: GT png, member pngs (titled with manuf status when
applicable), ensemble.npz (continuous fields, binaries, cond info, applied
manuf override), and printed mass / speckle / pairwise-IoU / manuf stats.

Requires v0720_neural_diff_trainer_fixed.py, v0723_masked_voxel_diff_trainer.py,
and (only if using --manuf_override_raw_value) v0804_cond_data_utils_neural_diff.py
in the same directory.
"""

import argparse
import functools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from v0720_neural_diff_trainer_fixed import (
    NeuralFieldDataset3D,
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    plot_voxel_with_overlays_implicit,
    decode_condition_overlay,
)
from v0723_masked_voxel_diff_trainer import (
    MaskedScoreNet3D,
    heun_sampler_masked,
    pad_to_multiple,
    speckle_fraction,
    pairwise_iou,
)


import re


def parse_bc_dofs_from_cond_str(cond_str):
    """
    Extracts BC degrees-of-freedom directly from cond_str, e.g.
    '..._bcD_(1,1,0)_(1,1,1)_(1,1,1)_load_...' -> [[1,1,0],[1,1,1],[1,1,1]].
    Self-contained: does not depend on decode_condition_overlay/cond_slices,
    so it works regardless of whether the shared v0720 dependency has the
    bc_dofs decoding patch applied. Returns None if no bcD_ segment found.
    """
    m = re.search(r"bcD_((?:\(\d,\d,\d\)_?)+)", str(cond_str))
    if not m:
        return None
    triples = re.findall(r"\((\d),(\d),(\d)\)", m.group(1))
    return [[int(a), int(b), int(c)] for a, b, c in triples]


def decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode):
    """Reads the manufacturability slot(s) straight out of a cond_vec, for
    printing/manifest purposes -- independent of whatever override logic
    was used to write them."""
    if manuf_mode == "omit":
        return {"has_label": None, "value": None, "label": None}

    def get(name):
        if name not in cond_slices:
            return None
        s, e = cond_slices[name]
        return float(cond_vec[s:e][0])

    hl = get("manuf_has_label")
    has_label = bool(round(hl)) if hl is not None else None
    value = get("manuf_value") if manuf_mode == "scalar" else None
    label = get("manuf_label") if manuf_mode == "binary" else None
    return {"has_label": has_label, "value": value, "label": label}


def compute_manuf_override_targets(manuf_mode, meta, args):
    """
    Resolves --manuf_override into the concrete (value, label) to write,
    validating mode-specific combinations up front so bad CLI combos fail
    fast with a clear message rather than silently writing nothing useful.
    Returns (target_value_or_None, target_label_or_None) -- exactly one is
    populated depending on manuf_mode, for override modes that need a value.
    """
    override = args["manuf_override"]
    if override == "none" or manuf_mode == "omit":
        return None, None

    threshold_pctl = float(meta["manuf_percentile_threshold"]) / 100.0  # e.g. 0.80

    if override == "empty":
        return None, None  # apply_manuf_override handles "empty" separately

    if override in ("positive", "negative"):
        if manuf_mode == "binary":
            return None, (1.0 if override == "positive" else -1.0)
        elif manuf_mode == "scalar":
            # No discrete label field in scalar mode -- use a representative
            # percentile on the correct side of the recorded split.
            if override == "positive":
                repr_pctl = threshold_pctl / 2.0
            else:
                repr_pctl = threshold_pctl + (1.0 - threshold_pctl) / 2.0
            print(f"  (scalar-mode {override} override -> representative "
                  f"percentile {repr_pctl:.4f}, split threshold is "
                  f"{threshold_pctl:.4f})")
            return repr_pctl, None

    if override == "value":
        if manuf_mode != "scalar":
            raise ValueError("--manuf_override value is only valid for "
                             "scalar-mode datasets; this dataset's "
                             "manufacturability_mode is "
                             f"{manuf_mode!r}. Use positive/negative/empty "
                             "instead.")
        if args["manuf_override_percentile"] is not None:
            return float(args["manuf_override_percentile"]), None
        if args["manuf_override_raw_value"] is not None:
            if not args["manufacturability_csv"]:
                raise ValueError("--manuf_override_raw_value requires "
                                 "--manufacturability_csv (to recompute the "
                                 "percentile rank against the same labeled "
                                 "population used at data-gen time).")
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from v0804_cond_data_utils_neural_diff import (
                load_manufacturability_scalar, compute_manufacturability_stats,
            )
            scalar_col = str(meta["manuf_scalar_column"])
            labeled_map = load_manufacturability_scalar(
                args["manufacturability_csv"], scalar_col)
            stats = compute_manufacturability_stats(
                labeled_map, float(meta["manuf_percentile_threshold"]))
            pctl = stats["percentile_rank_fn"](float(args["manuf_override_raw_value"]))
            print(f"  (raw value {args['manuf_override_raw_value']} -> "
                  f"percentile rank {pctl:.4f} against "
                  f"{stats['n_labeled_total']} labeled parts)")
            return pctl, None
        raise ValueError("--manuf_override value requires either "
                         "--manuf_override_percentile or "
                         "--manuf_override_raw_value (+ "
                         "--manufacturability_csv).")

    raise ValueError(f"Unhandled --manuf_override: {override}")


def apply_manuf_override(cond_vec, cond_slices, manuf_mode, override,
                         target_value=None, target_label=None):
    if manuf_mode == "omit" or override == "none":
        return cond_vec

    cond_vec = cond_vec.copy()

    def set_slice(name, val):
        s, e = cond_slices[name]
        cond_vec[s:e] = val

    if override == "empty":
        if manuf_mode == "scalar":
            set_slice("manuf_value", 0.0)
        elif manuf_mode == "binary":
            set_slice("manuf_label", 0.0)
        set_slice("manuf_has_label", 0.0)
        return cond_vec

    set_slice("manuf_has_label", 1.0)
    if manuf_mode == "scalar":
        set_slice("manuf_value", float(target_value))
    elif manuf_mode == "binary":
        set_slice("manuf_label", float(target_label))
    return cond_vec


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample ensembles from a trained masked voxel diffusion model")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to ckpt_best.pth (or any ckpt_*.pth)")
    p.add_argument("--hparams", type=str, default=None,
                   help="Path to hparams.json; default: <ckpt_dir>/../hparams.json")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)

    # which conditions
    p.add_argument("--indices", type=str, default="",
                   help="Comma-separated dataset indices, e.g. '3,17,102'. "
                        "Empty -> draw --n_random random conditions.")
    p.add_argument("--n_random", default=4, type=int)

    # sampling
    p.add_argument("--ensemble_size", default=8, type=int)
    p.add_argument("--n_steps", default=250, type=int)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    p.add_argument("--guidance_scale", default=0.0, type=float)

    # manufacturability override -- if the dataset's meta.npz records
    # manufacturability_mode != "omit", this lets you request a specific
    # manufacturability target for every sampled condition, overriding
    # whatever that dataset entry actually recorded (which may itself be
    # labeled, unlabeled, positive, or negative).
    p.add_argument("--manuf_override", type=str, default="none",
                   choices=["none", "positive", "negative", "empty", "value"],
                   help="none: leave the drawn condition's manufacturability "
                        "fields exactly as recorded in the dataset. "
                        "positive/negative: force a positive/negative label "
                        "(binary-mode datasets: writes +1/-1; scalar-mode "
                        "datasets: writes a representative percentile below/"
                        "above the recorded split threshold). "
                        "empty: force has_label=0, i.e. sample as if this "
                        "condition had no manufacturability information. "
                        "value: scalar-mode only -- write an exact target "
                        "via --manuf_override_percentile or "
                        "--manuf_override_raw_value.")
    p.add_argument("--manuf_override_percentile", type=float, default=None,
                   help="Scalar-mode + --manuf_override value: percentile "
                        "rank in (0,1] to write directly into the manuf_value "
                        "slot (same units the model was trained on).")
    p.add_argument("--manuf_override_raw_value", type=float, default=None,
                   help="Scalar-mode + --manuf_override value: a raw scalar "
                        "value (e.g. a deformation_p99 number) to convert to "
                        "a percentile rank via --manufacturability_csv before "
                        "writing into the manuf_value slot.")
    p.add_argument("--manufacturability_csv", type=str, default=None,
                   help="Required only if --manuf_override_raw_value is used: "
                        "path to the same summary CSV used at data-gen time, "
                        "so the raw value can be converted to a percentile "
                        "rank against the identical labeled population.")
    return vars(p.parse_args())


def main():
    args = parse_args()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    ckpt_path = Path(args["ckpt"])
    hparams_path = (Path(args["hparams"]) if args["hparams"] is not None
                    else ckpt_path.parent.parent / "hparams.json")
    with open(hparams_path) as f:
        hp = json.load(f)
    print(f"Loaded hparams from {hparams_path}")

    meta_path = args["meta_path"]
    if meta_path is None:
        meta_path = (hp.get("meta_path")
                     or args["data_file"][:-4] + "_meta.npz")
    meta = np.load(meta_path, allow_pickle=True)

    dataset = NeuralFieldDataset3D(args["data_file"])

    model = MaskedScoreNet3D(
        cond_dim=dataset.cond_dim,
        cond_embed_dim=hp["cond_embed_dim"],
        cond_ch=hp["cond_ch"],
        c1=hp["unet_ch1_dim"],
        c2=hp["unet_ch2_dim"],
        c3=hp["unet_ch3_dim"],
        ed=hp["t_embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    if args["guidance_scale"] > 0.0 and hp.get("cond_drop_prob", 0.0) <= 0.0:
        print("WARNING: guidance_scale > 0 but the model was trained with "
              "cond_drop_prob = 0; CFG is not valid for this checkpoint.")

    manuf_mode = str(meta["manufacturability_mode"]) if "manufacturability_mode" in meta else "omit"
    cond_slices = json.loads(str(meta["cond_slices_json"])) if "cond_slices_json" in meta else {}
    if manuf_mode != "omit":
        print(f"Dataset manufacturability_mode = {manuf_mode} "
              f"(threshold @ p{float(meta['manuf_percentile_threshold']):.0f} "
              f"= {float(meta['manuf_threshold_value']):.6g})")
    elif args["manuf_override"] != "none":
        print("WARNING: --manuf_override was set but this dataset's "
              "manufacturability_mode is 'omit' -- override will be ignored.")

    manuf_target_value, manuf_target_label = compute_manuf_override_targets(
        manuf_mode, meta, args)
    if args["manuf_override"] != "none" and manuf_mode != "omit":
        print(f"Applying manuf_override={args['manuf_override']!r} to every "
              f"sampled condition")

    if args["indices"].strip():
        chosen = [int(s) for s in args["indices"].split(",") if s.strip()]
    else:
        rng = np.random.default_rng(args["seed"])
        chosen = rng.choice(len(dataset), size=args["n_random"],
                            replace=False).tolist()
    print(f"Sampling conditions (dataset indices): {chosen}")

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    outdir = Path(args["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    E = args["ensemble_size"]

    for ci, idx in enumerate(chosen):
        idx = int(idx)
        vox = dataset.voxels[idx]
        Nx, Ny, Nz = vox.shape
        Px, Py, Pz = (pad_to_multiple(Nx), pad_to_multiple(Ny),
                      pad_to_multiple(Nz))
        mask = torch.zeros(E, 1, Px, Py, Pz)
        mask[:, :, :Nx, :Ny, :Nz] = 1.0
        cond_vec = dataset.conds[idx]
        original_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
        cond_vec = apply_manuf_override(
            cond_vec, cond_slices, manuf_mode, args["manuf_override"],
            target_value=manuf_target_value, target_label=manuf_target_label,
        )
        applied_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
        cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)

        gen = torch.Generator(device=device)
        gen.manual_seed(args["seed"] * 1000 + ci)
        x_final = heun_sampler_masked(
            model, mask, cond, mean_fn, std_fn, drift_fn,
            n_steps=args["n_steps"], eps=args["sample_eps"], device=device,
            guidance_scale=args["guidance_scale"], generator=gen,
        ).numpy()[:, 0, :Nx, :Ny, :Nz]

        bins = [(x_final[e] > 0.0).astype(np.float32) for e in range(E)]
        masses = [float(b.mean()) for b in bins]
        speckles = [speckle_fraction(b) for b in bins]
        iou = pairwise_iou(bins)
        gt_mass = float(vox.mean())

        overlay = decode_condition_overlay(cond_vec, meta)
        cdir = outdir / f"cond{ci:02d}_idx{idx}"
        cdir.mkdir(exist_ok=True)

        def _manuf_title():
            if manuf_mode == "omit":
                return ""
            if not applied_manuf["has_label"]:
                return " | manuf: UNLABELED"
            if manuf_mode == "scalar":
                return f" | manuf: value={applied_manuf['value']:.4f}"
            return f" | manuf: {'POSITIVE' if applied_manuf['label'] > 0 else 'NEGATIVE'}"

        manuf_title_suffix = _manuf_title()

        for e in range(E):
            plot_voxel_with_overlays_implicit(
                bins[e], overlay["bc_points"], overlay["load_point"],
                overlay["load_dir"],
                title=(f"idx {idx} member {e} | ({Nx},{Ny},{Nz}) | "
                       f"mass {masses[e]:.3f} | speckle {speckles[e]:.3f}"
                       f"{manuf_title_suffix}"),
                save_path=cdir / f"member{e:02d}.png",
            )
        plot_voxel_with_overlays_implicit(
            vox, overlay["bc_points"], overlay["load_point"],
            overlay["load_dir"],
            title=f"GT idx {idx} | ({Nx},{Ny},{Nz}) | mass {gt_mass:.3f}",
            save_path=cdir / "gt.png",
        )
        sample_info_for_npz = dataset.sample_infos[idx]
        source_index_for_npz = (sample_info_for_npz.get("source_index")
                                if isinstance(sample_info_for_npz, dict) else None)
        np.savez_compressed(
            cdir / "ensemble.npz",
            fields=x_final, binaries=np.stack(bins),
            cond=cond_vec, gt=vox, dataset_index=idx,
            source_index=source_index_for_npz,
            cond_str=str(dataset.cond_strs[idx]),
            bc_dofs=parse_bc_dofs_from_cond_str(dataset.cond_strs[idx]),
            guidance_scale=args["guidance_scale"],
            manuf_mode=manuf_mode,
            manuf_override=args["manuf_override"],
        )

        # Human-readable provenance + decoded conditions. Coordinates are in
        # the dataset convention: [0,1], normalized by max(Nx,Ny,Nz); multiply
        # by (max_dim - 1) for voxel indices.
        def _tolist(a):
            return None if a is None else np.asarray(a, dtype=float).tolist()

        sample_info = dataset.sample_infos[idx]
        source_index = (sample_info.get("source_index")
                        if isinstance(sample_info, dict) else None)
        bc_dofs_parsed = parse_bc_dofs_from_cond_str(dataset.cond_strs[idx])
        manifest = {
            "provenance": {
                "dataset_index": idx,
                "source_index": source_index,
                "data_file": str(args["data_file"]),
                "meta_path": str(meta_path),
                "ckpt": str(ckpt_path),
                "generated": datetime.now().isoformat(timespec="seconds"),
            },
            "part": {
                "shape": [Nx, Ny, Nz],
                "gt_mass_fraction": gt_mass,
                "cond_str": str(dataset.cond_strs[idx]),
                "sample_info": {k: (v.tolist() if isinstance(v, np.ndarray)
                                    else v)
                                for k, v in dict(sample_info).items()}
                               if isinstance(sample_info, dict) else None,
            },
            "conditions_decoded": {
                "coordinate_convention":
                    "[0,1] normalized by max(Nx,Ny,Nz); "
                    "voxel index ~= coord * (max_dim - 1)",
                "bc_points": _tolist(overlay["bc_points"]),
                "bc_dofs": bc_dofs_parsed,
                "load_point": _tolist(overlay["load_point"]),
                "load_dir": _tolist(overlay["load_dir"]),
                "conditioning_spec": overlay["conditioning_spec"],
                "cond_vector": cond_vec.astype(float).tolist(),
            },
            "manufacturability": {
                "mode": manuf_mode,
                "override_requested": args["manuf_override"],
                "as_originally_drawn": original_manuf,
                "as_applied_to_this_sample": applied_manuf,
            },
            "sampling": {
                "ensemble_size": E,
                "n_steps": args["n_steps"],
                "sample_eps": args["sample_eps"],
                "guidance_scale": args["guidance_scale"],
                "seed": args["seed"],
                "noise_seed": args["seed"] * 1000 + ci,
            },
            "results": {
                "member_mass_fractions": [round(m, 5) for m in masses],
                "member_speckle_fractions": [round(s, 5) for s in speckles],
                "pairwise_iou": round(iou, 5) if iou == iou else None,
            },
        }
        with open(cdir / "manifest.json", "w") as jf:
            json.dump(manifest, jf, indent=2, default=str)

        print(f"[cond {ci}] idx {idx} (source_index {source_index}) "
              f"shape ({Nx},{Ny},{Nz})  GT mass {gt_mass:.3f}")
        print(f"          member masses  {[round(m,3) for m in masses]}")
        print(f"          speckle fracs  {[round(s,3) for s in speckles]}  "
              f"(GT ~ 0.00-0.02; target < 0.05)")
        print(f"          pairwise IoU   {iou:.3f}  (target 0.5-0.8)")
        if manuf_mode != "omit":
            print(f"          manuf (drawn)   {original_manuf}")
            print(f"          manuf (applied) {applied_manuf}")

    print(f"Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()