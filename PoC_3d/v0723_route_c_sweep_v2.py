"""
Route C hyperparameter sweep: guidance_scale x n_steps.

v2: adds a BC/load CONTACT metric -- for each condition, checks whether
each BC point and the load point have solid material within a small voxel
radius, which mass/speckle/IoU cannot see. This is what motivated the v2
update: mass error bottomed out around w=2 in the first sweep, but visual
inspection showed BC points still floating off the surface at w=2 and
looking better at w=4 -- none of the original three metrics could confirm
or refute that, so this version measures it directly instead of relying on
single-frame visual inspection.

Also: saves a preview image for EVERY fixed condition per grid point (not
just the first), since a single preview frame isn't enough to judge contact
reliably across the sweep.

Runs the SAME sampling path as v0723_masked_voxel_diff_sampler.py (imports
its functions directly, no reimplementation), on a FIXED set of
conditions/seeds so results are directly comparable across grid points.

Outputs (under --outdir):
  sweep_results.csv      one row per (guidance_scale, n_steps) grid point,
                          now including mean_bc_contact / mean_load_contact
  sweep_manifest.json    full per-condition, per-grid-point results
  gt_contact_check.json  GT contact rates per condition -- a calibration
                          check: GT should score ~1.0 contact at the chosen
                          radius, since it's ground truth by construction.
                          If GT itself doesn't score ~1.0, --contact_radius
                          is too tight and needs widening before trusting
                          the generated-sample numbers.
  previews/gscaleW_stepsN_idxIDX.png   one quick-look image per grid point
                                        PER fixed condition

Usage (same required args as the standalone sampler, plus the grid):

  python v0723_route_c_sweep.py \
      --ckpt <run_dir>/checkpoints/ckpt_best.pth \
      --data_file <path>/10000_neuralfield_..._to_condition.npy \
      --outdir sweep_out \
      --guidance_scales 0,0.5,1,2,3,4,6 \
      --n_steps_list 250 \
      --n_random 4 --ensemble_size 8 --contact_radius 1

Requires v0720_neural_diff_trainer_fixed.py and
v0723_masked_voxel_diff_trainer.py in the same directory (same requirement
as the standalone sampler).
"""

import argparse
import functools
import json
import csv
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


def parse_args():
    p = argparse.ArgumentParser(description="Route C guidance_scale x n_steps sweep")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hparams", type=str, default=None)
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)

    p.add_argument("--indices", type=str, default="",
                   help="Comma-separated dataset indices to hold fixed across "
                        "the whole sweep. Empty -> draw --n_random once, up "
                        "front, and reuse for every grid point.")
    p.add_argument("--n_random", default=4, type=int)

    p.add_argument("--ensemble_size", default=8, type=int)
    p.add_argument("--sample_eps", default=1e-3, type=float)

    p.add_argument("--guidance_scales", type=str, default="0,0.5,1,2,3,4,6",
                   help="Comma-separated guidance_scale values to sweep.")
    p.add_argument("--n_steps_list", type=str, default="250",
                   help="Comma-separated n_steps values to sweep.")
    p.add_argument("--contact_radius", type=int, default=1,
                   help="Chebyshev voxel radius used to decide whether a "
                        "BC/load point 'touches' solid material -- a "
                        "(2r+1)^3 cube neighborhood around the rounded "
                        "voxel index is checked for any solid voxel.")
    return vars(p.parse_args())


def voxel_contact_fraction(binary_vox, points01, radius=1):
    """
    Fraction of points01 (K,3), each in [0,1] isotropic-normalized coords,
    that have >=1 solid voxel within a Chebyshev `radius` neighborhood in
    binary_vox. Returns None if points01 is empty/None.
    """
    if points01 is None:
        return None
    points01 = np.atleast_2d(points01)
    if points01.size == 0:
        return None
    Nx, Ny, Nz = binary_vox.shape
    c = float(max(Nx, Ny, Nz))
    hits = 0
    for p in points01:
        ix = int(round(float(p[0]) * (c - 1.0)))
        iy = int(round(float(p[1]) * (c - 1.0)))
        iz = int(round(float(p[2]) * (c - 1.0)))
        ix = min(max(ix, 0), Nx - 1)
        iy = min(max(iy, 0), Ny - 1)
        iz = min(max(iz, 0), Nz - 1)
        x0, x1 = max(0, ix - radius), min(Nx, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(Ny, iy + radius + 1)
        z0, z1 = max(0, iz - radius), min(Nz, iz + radius + 1)
        if binary_vox[x0:x1, y0:y1, z0:z1].sum() > 0:
            hits += 1
    return hits / len(points01)


def main():
    args = parse_args()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    R = args["contact_radius"]

    ckpt_path = Path(args["ckpt"])
    hparams_path = (Path(args["hparams"]) if args["hparams"] is not None
                    else ckpt_path.parent.parent / "hparams.json")
    with open(hparams_path) as f:
        hp = json.load(f)
    print(f"Loaded hparams from {hparams_path}")

    meta_path = args["meta_path"] or hp.get("meta_path") or args["data_file"][:-4] + "_meta.npz"
    meta = np.load(meta_path, allow_pickle=True)

    dataset = NeuralFieldDataset3D(args["data_file"])

    model = MaskedScoreNet3D(
        cond_dim=dataset.cond_dim,
        cond_embed_dim=hp["cond_embed_dim"],
        cond_ch=hp["cond_ch"],
        c1=hp["unet_ch1_dim"], c2=hp["unet_ch2_dim"], c3=hp["unet_ch3_dim"],
        ed=hp["t_embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    if hp.get("cond_drop_prob", 0.0) <= 0.0:
        print("WARNING: model trained with cond_drop_prob=0; guidance_scale>0 "
              "results below are not valid CFG for this checkpoint.")

    # Fixed condition set for the whole sweep.
    if args["indices"].strip():
        chosen = [int(s) for s in args["indices"].split(",") if s.strip()]
    else:
        rng = np.random.default_rng(args["seed"])
        chosen = rng.choice(len(dataset), size=args["n_random"], replace=False).tolist()
    print(f"Fixed conditions (dataset indices) for the whole sweep: {chosen}")

    gscales = [float(s) for s in args["guidance_scales"].split(",") if s.strip() != ""]
    steps_list = [int(s) for s in args["n_steps_list"].split(",") if s.strip() != ""]

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    outdir = Path(args["outdir"])
    (outdir / "previews").mkdir(parents=True, exist_ok=True)
    E = args["ensemble_size"]

    # ---- GT contact calibration check (independent of w / n_steps) ----
    gt_contact_check = {}
    print(f"\nGT contact calibration (radius={R} voxels) -- should be ~1.0:")
    overlays_by_idx = {}
    for idx in chosen:
        idx = int(idx)
        vox = dataset.voxels[idx].astype(np.float32)
        cond_vec = dataset.conds[idx]
        overlay = decode_condition_overlay(cond_vec, meta)
        overlays_by_idx[idx] = overlay
        bc_c = voxel_contact_fraction(vox, overlay["bc_points"], radius=R)
        load_c = voxel_contact_fraction(
            vox, overlay["load_point"][None] if overlay["load_point"] is not None else None,
            radius=R)
        gt_contact_check[idx] = {"gt_bc_contact": bc_c, "gt_load_contact": load_c}
        print(f"  idx {idx}: bc_contact={bc_c}  load_contact={load_c}")
    with open(outdir / "gt_contact_check.json", "w") as f:
        json.dump(gt_contact_check, f, indent=2)
    low_gt = [i for i, v in gt_contact_check.items()
              if (v["gt_bc_contact"] or 1.0) < 0.99 or (v["gt_load_contact"] or 1.0) < 0.99]
    if low_gt:
        print(f"  WARNING: GT contact < 1.0 for conditions {low_gt} at radius={R}. "
              f"Consider increasing --contact_radius before trusting the sweep numbers below.")
    print()

    csv_rows = []
    full_manifest = {"grid": [], "conditions": chosen, "contact_radius": R}

    for w in gscales:
        for n_steps in steps_list:
            per_cond_masses, per_cond_speckles, per_cond_ious = [], [], []
            per_cond_bc_contact, per_cond_load_contact = [], []

            for ci, idx in enumerate(chosen):
                idx = int(idx)
                vox = dataset.voxels[idx]
                Nx, Ny, Nz = vox.shape
                Px, Py, Pz = (pad_to_multiple(Nx), pad_to_multiple(Ny), pad_to_multiple(Nz))
                mask = torch.zeros(E, 1, Px, Py, Pz)
                mask[:, :, :Nx, :Ny, :Nz] = 1.0
                cond_vec = dataset.conds[idx]
                cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)
                overlay = overlays_by_idx[idx]

                # Same noise seed for a given condition across every grid
                # point -> differences you see are due to (w, n_steps) only.
                gen = torch.Generator(device=device)
                gen.manual_seed(args["seed"] * 1000 + ci)

                x_final = heun_sampler_masked(
                    model, mask, cond, mean_fn, std_fn, drift_fn,
                    n_steps=n_steps, eps=args["sample_eps"], device=device,
                    guidance_scale=w, generator=gen,
                ).numpy()[:, 0, :Nx, :Ny, :Nz]

                bins = [(x_final[e] > 0.0).astype(np.float32) for e in range(E)]
                gt_mass = float(vox.mean())
                masses = [float(b.mean()) for b in bins]
                speckles = [speckle_fraction(b) for b in bins]
                iou = pairwise_iou(bins)

                bc_contacts = [voxel_contact_fraction(b, overlay["bc_points"], radius=R)
                              for b in bins]
                bc_contacts = [c for c in bc_contacts if c is not None]
                load_contacts = [voxel_contact_fraction(
                    b, overlay["load_point"][None] if overlay["load_point"] is not None else None,
                    radius=R) for b in bins]
                load_contacts = [c for c in load_contacts if c is not None]

                per_cond_masses.append(np.mean([abs(m - gt_mass) for m in masses]))
                per_cond_speckles.append(float(np.mean(speckles)))
                if iou == iou:  # skip NaN (E<2 case)
                    per_cond_ious.append(iou)
                if bc_contacts:
                    per_cond_bc_contact.append(float(np.mean(bc_contacts)))
                if load_contacts:
                    per_cond_load_contact.append(float(np.mean(load_contacts)))

                plot_voxel_with_overlays_implicit(
                    bins[0], overlay["bc_points"], overlay["load_point"],
                    overlay["load_dir"],
                    title=(f"idx {idx} | w={w} steps={n_steps} | "
                           f"mass {masses[0]:.3f} bc_contact "
                           f"{bc_contacts[0] if bc_contacts else 'n/a'}"),
                    save_path=outdir / "previews" / f"gscale{w}_steps{n_steps}_idx{idx}.png",
                )

                full_manifest["grid"].append({
                    "guidance_scale": w, "n_steps": n_steps,
                    "condition_index": idx,
                    "gt_mass": gt_mass,
                    "member_masses": [round(m, 5) for m in masses],
                    "member_speckles": [round(s, 5) for s in speckles],
                    "pairwise_iou": round(iou, 5) if iou == iou else None,
                    "member_bc_contact": [round(c, 5) for c in bc_contacts],
                    "member_load_contact": [round(c, 5) for c in load_contacts],
                })

            row = {
                "guidance_scale": w,
                "n_steps": n_steps,
                "mean_abs_mass_error": round(float(np.mean(per_cond_masses)), 5),
                "mean_speckle": round(float(np.mean(per_cond_speckles)), 5),
                "mean_pairwise_iou": (round(float(np.mean(per_cond_ious)), 5)
                                      if per_cond_ious else None),
                "mean_bc_contact": (round(float(np.mean(per_cond_bc_contact)), 5)
                                    if per_cond_bc_contact else None),
                "mean_load_contact": (round(float(np.mean(per_cond_load_contact)), 5)
                                      if per_cond_load_contact else None),
            }
            csv_rows.append(row)
            print(f"[w={w:>4} steps={n_steps:>4}] "
                  f"mass_err={row['mean_abs_mass_error']:.4f}  "
                  f"speckle={row['mean_speckle']:.4f}  "
                  f"iou={row['mean_pairwise_iou']}  "
                  f"bc_contact={row['mean_bc_contact']}  "
                  f"load_contact={row['mean_load_contact']}")

    with open(outdir / "sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(outdir / "sweep_manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2, default=str)

    print(f"\nDone. Summary: {outdir/'sweep_results.csv'}")
    print(f"Full per-condition detail: {outdir/'sweep_manifest.json'}")
    print(f"GT contact calibration: {outdir/'gt_contact_check.json'}")
    print(f"Quick-look previews (per condition, per grid point): {outdir/'previews'}/")


if __name__ == "__main__":
    main()