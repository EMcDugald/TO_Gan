# """
# Standalone ensemble sampler for the Route C masked voxel diffusion model.

# Loads a checkpoint + its hparams.json (auto-inferred from the ckpt dir),
# picks conditions from the dataset (specific indices or seeded random draws),
# and generates an ensemble per condition with the batched Heun sampler.

#   --guidance_scale w   classifier-free guidance (valid because the trainer
#                        drops the condition with prob cond_drop_prob):
#                          x0hat = (1+w)*x0hat_cond - w*x0hat_uncond
#                        w=0 -> plain conditional sampling (max diversity).

# Outputs per condition: GT png, member pngs, ensemble.npz (continuous fields,
# binaries, cond info), and printed mass / speckle / pairwise-IoU stats --
# the same metrics as the v0722 diagnostics, so runs are directly comparable.

# Requires v0720_neural_diff_trainer_fixed.py and
# v0723_masked_voxel_diff_trainer.py in the same directory.
# """

# import argparse
# import functools
# import json
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
#         for e in range(E):
#             plot_voxel_with_overlays_implicit(
#                 bins[e], overlay["bc_points"], overlay["load_point"],
#                 overlay["load_dir"],
#                 title=(f"idx {idx} member {e} | ({Nx},{Ny},{Nz}) | "
#                        f"mass {masses[e]:.3f} | speckle {speckles[e]:.3f}"),
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
#         )

#         print(f"[cond {ci}] idx {idx} shape ({Nx},{Ny},{Nz})  "
#               f"GT mass {gt_mass:.3f}")
#         print(f"          member masses  {[round(m,3) for m in masses]}")
#         print(f"          speckle fracs  {[round(s,3) for s in speckles]}  "
#               f"(GT ~ 0.00-0.02; target < 0.05)")
#         print(f"          pairwise IoU   {iou:.3f}  (target 0.5-0.8)")

#     print(f"Done. Outputs in {outdir}")


# if __name__ == "__main__":
#     main()


"""
Standalone ensemble sampler for the Route C masked voxel diffusion model.

Loads a checkpoint + its hparams.json (auto-inferred from the ckpt dir),
picks conditions from the dataset (specific indices or seeded random draws),
and generates an ensemble per condition with the batched Heun sampler.

  --guidance_scale w   classifier-free guidance (valid because the trainer
                       drops the condition with prob cond_drop_prob):
                         x0hat = (1+w)*x0hat_cond - w*x0hat_uncond
                       w=0 -> plain conditional sampling (max diversity).

Outputs per condition: GT png, member pngs, ensemble.npz (continuous fields,
binaries, cond info), and printed mass / speckle / pairwise-IoU stats --
the same metrics as the v0722 diagnostics, so runs are directly comparable.

Requires v0720_neural_diff_trainer_fixed.py and
v0723_masked_voxel_diff_trainer.py in the same directory.
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
        for e in range(E):
            plot_voxel_with_overlays_implicit(
                bins[e], overlay["bc_points"], overlay["load_point"],
                overlay["load_dir"],
                title=(f"idx {idx} member {e} | ({Nx},{Ny},{Nz}) | "
                       f"mass {masses[e]:.3f} | speckle {speckles[e]:.3f}"),
                save_path=cdir / f"member{e:02d}.png",
            )
        plot_voxel_with_overlays_implicit(
            vox, overlay["bc_points"], overlay["load_point"],
            overlay["load_dir"],
            title=f"GT idx {idx} | ({Nx},{Ny},{Nz}) | mass {gt_mass:.3f}",
            save_path=cdir / "gt.png",
        )
        np.savez_compressed(
            cdir / "ensemble.npz",
            fields=x_final, binaries=np.stack(bins),
            cond=cond_vec, gt=vox, dataset_index=idx,
            cond_str=str(dataset.cond_strs[idx]),
            guidance_scale=args["guidance_scale"],
        )

        # Human-readable provenance + decoded conditions. Coordinates are in
        # the dataset convention: [0,1], normalized by max(Nx,Ny,Nz); multiply
        # by (max_dim - 1) for voxel indices.
        def _tolist(a):
            return None if a is None else np.asarray(a, dtype=float).tolist()

        sample_info = dataset.sample_infos[idx]
        manifest = {
            "provenance": {
                "dataset_index": idx,
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
                "load_point": _tolist(overlay["load_point"]),
                "load_dir": _tolist(overlay["load_dir"]),
                "conditioning_spec": overlay["conditioning_spec"],
                "cond_vector": cond_vec.astype(float).tolist(),
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

        print(f"[cond {ci}] idx {idx} shape ({Nx},{Ny},{Nz})  "
              f"GT mass {gt_mass:.3f}")
        print(f"          member masses  {[round(m,3) for m in masses]}")
        print(f"          speckle fracs  {[round(s,3) for s in speckles]}  "
              f"(GT ~ 0.00-0.02; target < 0.05)")
        print(f"          pairwise IoU   {iou:.3f}  (target 0.5-0.8)")

    print(f"Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()