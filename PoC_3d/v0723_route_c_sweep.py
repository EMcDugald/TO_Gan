"""
Route C hyperparameter sweep: guidance_scale x n_steps.

Runs the SAME sampling path as v0723_masked_voxel_diff_sampler.py (imports
its functions directly, no reimplementation), but loops over a grid of
(guidance_scale, n_steps) on a FIXED set of conditions/seeds so results are
directly comparable frame-by-frame and number-by-number.

For each grid point, generates an ensemble per condition, computes
mass / speckle / pairwise-IoU averaged across conditions, and saves one
representative member image (condition 0, member 0) so you can eyeball the
guidance/step trade-off without opening every folder.

Outputs (under --outdir):
  sweep_results.csv      one row per (guidance_scale, n_steps) grid point
  sweep_manifest.json    full per-condition, per-grid-point results
  previews/gscaleW_stepsN.png   quick-look image per grid point

Usage (same required args as the standalone sampler, plus the grid):

  python v0723_route_c_sweep.py \
      --ckpt <run_dir>/checkpoints/ckpt_best.pth \
      --data_file <path>/10000_neuralfield_..._to_condition.npy \
      --outdir sweep_out \
      --guidance_scales 0,0.5,1,2,4 \
      --n_steps_list 150,250 \
      --n_random 4 --ensemble_size 8

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

    p.add_argument("--guidance_scales", type=str, default="0,0.5,1,2,4",
                   help="Comma-separated guidance_scale values to sweep.")
    p.add_argument("--n_steps_list", type=str, default="250",
                   help="Comma-separated n_steps values to sweep.")
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

    csv_rows = []
    full_manifest = {"grid": [], "conditions": chosen}

    for w in gscales:
        for n_steps in steps_list:
            per_cond_masses, per_cond_speckles, per_cond_ious = [], [], []
            first_member_img = None

            for ci, idx in enumerate(chosen):
                idx = int(idx)
                vox = dataset.voxels[idx]
                Nx, Ny, Nz = vox.shape
                Px, Py, Pz = (pad_to_multiple(Nx), pad_to_multiple(Ny), pad_to_multiple(Nz))
                mask = torch.zeros(E, 1, Px, Py, Pz)
                mask[:, :, :Nx, :Ny, :Nz] = 1.0
                cond_vec = dataset.conds[idx]
                cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)

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

                per_cond_masses.append(np.mean([abs(m - gt_mass) for m in masses]))
                per_cond_speckles.append(float(np.mean(speckles)))
                if iou == iou:  # skip NaN (E<2 case)
                    per_cond_ious.append(iou)

                if ci == 0:
                    overlay = decode_condition_overlay(cond_vec, meta)
                    first_member_img = (bins[0], overlay, idx, Nx, Ny, Nz, masses[0], speckles[0])

                full_manifest["grid"].append({
                    "guidance_scale": w, "n_steps": n_steps,
                    "condition_index": idx,
                    "gt_mass": gt_mass,
                    "member_masses": [round(m, 5) for m in masses],
                    "member_speckles": [round(s, 5) for s in speckles],
                    "pairwise_iou": round(iou, 5) if iou == iou else None,
                })

            row = {
                "guidance_scale": w,
                "n_steps": n_steps,
                "mean_abs_mass_error": round(float(np.mean(per_cond_masses)), 5),
                "mean_speckle": round(float(np.mean(per_cond_speckles)), 5),
                "mean_pairwise_iou": (round(float(np.mean(per_cond_ious)), 5)
                                      if per_cond_ious else None),
            }
            csv_rows.append(row)
            print(f"[w={w:>4} steps={n_steps:>4}] "
                  f"mass_err={row['mean_abs_mass_error']:.4f}  "
                  f"speckle={row['mean_speckle']:.4f}  "
                  f"iou={row['mean_pairwise_iou']}")

            if first_member_img is not None:
                b0, overlay, idx, Nx, Ny, Nz, m0, s0 = first_member_img
                plot_voxel_with_overlays_implicit(
                    b0, overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
                    title=(f"idx {idx} | w={w} steps={n_steps} | "
                           f"mass {m0:.3f} speckle {s0:.3f}"),
                    save_path=outdir / "previews" / f"gscale{w}_steps{n_steps}.png",
                )

    with open(outdir / "sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(outdir / "sweep_manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2, default=str)

    print(f"\nDone. Summary: {outdir/'sweep_results.csv'}")
    print(f"Full per-condition detail: {outdir/'sweep_manifest.json'}")
    print(f"Quick-look previews: {outdir/'previews'}/")


if __name__ == "__main__":
    main()