"""
Standalone ensemble sampler for the Route C masked voxel diffusion model,
v1001: adds inpainting-style SOLID PINNING at the load point (and optionally
the BC points), plus reproducible condition selection.

Built as a standalone copy of v0804_masked_voxel_diff_sampler.py so the
existing sampler stays untouched while this method is tested. With
--pin_load off, --pin_bc off and default index options, it reproduces the
v0804 sampler's behavior exactly (same index draw, same noise seeds, same
solver), so pinned vs. unpinned runs are a clean A/B.

Changelog vs v0804_masked_voxel_diff_sampler.py
-----------------------------------------------
1. --pin_load / --pin_bc / --pin_radius
   Inpainting by replacement in x0-space, the mirror image of the existing
   padding pin. Every solver step, after CFG is combined, the model's x0
   prediction is overwritten:
       padding                 -> -1 (void)   [unchanged from v0804]
       pin cube around point   -> +1 (solid)  [new]
   The pin cube is the (2r+1)^3 voxel neighborhood (clipped to the part's
   true extent) around the point's voxel index, using the same
   coord -> index map as the validated contact metric
   (round(coord * (max(Nx,Ny,Nz) - 1)), clipped). Default r = 1 matches
   the radius at which GT parts scored load contact 1.0 in the v0723
   contact sweep. The network's inputs are NOT changed (it never saw a pin
   channel in training); the pin acts only through the x0 replacement, so
   the network sees solid material appear in its noisy input x_t over the
   trajectory and conditions the surrounding structure on it.
   After the last step, any pinned voxel still <= 0 is forced to +1 and
   COUNTED (n_pin_voxels_forced_at_end in the manifest). That count is a
   loud flag: nonzero means the trajectory alone did not honor the pin.

2. Connectivity diagnostic (per member): is the pinned material part of the
   largest 6-connected solid component? Pinning can in principle produce a
   floating blob at the load point that never joins the structure; this
   flag catches that. Recorded as pin_in_largest_component, plus
   pin_component_frac (size of the component touching the pin divided by
   total solid). Uses scipy.ndimage (already a dependency of v0720).

3. Load / BC contact metric (radius --contact_radius, default 1), reported
   per member for both pinned and unpinned runs, so a PIN_LOAD=0 run gives
   the baseline to compare against.

4. Reproducible condition selection
   * --indices_from_run DIR: reuse exactly the conditions (same order) of
     an earlier run, parsed from its cond##_idx#### folder names.
   * chosen_indices.json is written to every outdir (dataset_index list,
     source_index list, selection mode), so any run can be repeated.
   * --val_only: restrict to the held-out validation split, reproduced
     from the checkpoint's hparams.json (seed, nsamples, val_frac) using
     the trainer's exact split code.
   * --noise_seed_mode {position,index}: "position" (default, v0804
     behavior) seeds member noise by the condition's position in the list;
     "index" seeds by dataset index, so a condition gets the same noise
     regardless of which list it appears in or where.

5. bc_dofs and source_index come from cond_str / sample_info directly
   (unchanged from the patched v0804), so the cluster's unpatched v0720
   decode_condition_overlay is fine.

Requires in the same directory: v0720_neural_diff_trainer_fixed.py,
v0723_masked_voxel_diff_trainer.py, and (only for --manuf_override_raw_value)
v0804_cond_data_utils_neural_diff.py.
"""

import argparse
import functools
import json
import re
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
    pad_to_multiple,
    speckle_fraction,
    pairwise_iou,
)


# --------------------------------------------------------------------------
# condition-string / provenance helpers (unchanged from v0804)
# --------------------------------------------------------------------------

def parse_bc_dofs_from_cond_str(cond_str):
    """'..._bcD_(1,1,0)_(1,1,1)_load_...' -> [[1,1,0],[1,1,1]]; None if absent."""
    m = re.search(r"bcD_((?:\(\d,\d,\d\)_?)+)", str(cond_str))
    if not m:
        return None
    triples = re.findall(r"\((\d),(\d),(\d)\)", m.group(1))
    return [[int(a), int(b), int(c)] for a, b, c in triples]


def _source_index(sample_info):
    return sample_info.get("source_index") if isinstance(sample_info, dict) else None


# --------------------------------------------------------------------------
# manufacturability override (unchanged from v0804)
# --------------------------------------------------------------------------

def decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode):
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
    override = args["manuf_override"]
    if override == "none" or manuf_mode == "omit":
        return None, None

    threshold_pctl = float(meta["manuf_percentile_threshold"]) / 100.0

    if override == "empty":
        return None, None

    if override in ("positive", "negative"):
        if manuf_mode == "binary":
            return None, (1.0 if override == "positive" else -1.0)
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
            raise ValueError("--manuf_override value is only valid for scalar-mode "
                             f"datasets; this dataset is {manuf_mode!r}. Use "
                             "positive/negative/empty instead.")
        if args["manuf_override_percentile"] is not None:
            pct = float(args["manuf_override_percentile"])
            if not (0.0 < pct <= 1.0):
                print(f"  WARNING: --manuf_override_percentile {pct} is outside "
                      "(0,1], the range the model was trained on. Proceeding, but "
                      "this is extrapolation.")
            return pct, None
        if args["manuf_override_raw_value"] is not None:
            if not args["manufacturability_csv"]:
                raise ValueError("--manuf_override_raw_value requires "
                                 "--manufacturability_csv.")
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from v0804_cond_data_utils_neural_diff import (
                load_manufacturability_scalar, compute_manufacturability_stats,
            )
            labeled_map = load_manufacturability_scalar(
                args["manufacturability_csv"], str(meta["manuf_scalar_column"]))
            stats = compute_manufacturability_stats(
                labeled_map, float(meta["manuf_percentile_threshold"]))
            pctl = stats["percentile_rank_fn"](float(args["manuf_override_raw_value"]))
            print(f"  (raw value {args['manuf_override_raw_value']} -> percentile "
                  f"rank {pctl:.4f} against {stats['n_labeled_total']} labeled parts)")
            return pctl, None
        raise ValueError("--manuf_override value requires --manuf_override_percentile "
                         "or --manuf_override_raw_value (+ --manufacturability_csv).")

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
        set_slice("manuf_value" if manuf_mode == "scalar" else "manuf_label", 0.0)
        set_slice("manuf_has_label", 0.0)
        return cond_vec

    set_slice("manuf_has_label", 1.0)
    if manuf_mode == "scalar":
        set_slice("manuf_value", float(target_value))
    else:
        set_slice("manuf_label", float(target_label))
    return cond_vec


# --------------------------------------------------------------------------
# NEW: solid pinning
# --------------------------------------------------------------------------

def point01_to_voxel_index(p, true_shape):
    """[0,1]-by-max-dim coords -> clipped voxel index. Same map as the
    contact metric validated in v0723_route_c_sweep_v2 (GT load contact 1.0)."""
    Nx, Ny, Nz = true_shape
    c = float(max(Nx, Ny, Nz))
    idx = []
    for k, n in enumerate((Nx, Ny, Nz)):
        i = int(round(float(p[k]) * (c - 1.0)))
        idx.append(min(max(i, 0), n - 1))
    return tuple(idx)


def build_solid_pin_mask(true_shape, padded_shape, points01, radius):
    """
    Boolean array of padded_shape: True on the (2r+1)^3 cube (clipped to the
    true extent) around each point. Returns (mask, list_of_center_indices).
    """
    pin = np.zeros(padded_shape, dtype=bool)
    centers = []
    if points01 is None:
        return pin, centers
    pts = np.atleast_2d(np.asarray(points01, dtype=np.float64))
    if pts.size == 0:
        return pin, centers
    Nx, Ny, Nz = true_shape
    for p in pts:
        ix, iy, iz = point01_to_voxel_index(p, true_shape)
        centers.append([ix, iy, iz])
        pin[max(0, ix - radius):min(Nx, ix + radius + 1),
            max(0, iy - radius):min(Ny, iy + radius + 1),
            max(0, iz - radius):min(Nz, iz + radius + 1)] = True
    return pin, centers


def heun_sampler_masked_pinned(model, mask, cond, mean_fn, std_fn, drift_fn,
                               n_steps=150, eps=1e-3, device="cuda",
                               guidance_scale=0.0, generator=None,
                               solid_pin=None):
    """
    Copy of v0723 heun_sampler_masked with one addition: solid_pin
    ([B,1,D,H,W] in {0,1}, subset of the domain) forces x0hat to +1 there,
    every step, after CFG and after the padding pin. With solid_pin=None
    this is numerically identical to heun_sampler_masked.
    """
    model.eval()
    B = mask.shape[0]
    mask = mask.to(device)
    cond = cond.to(device)
    pin = solid_pin.to(device) if solid_pin is not None else None
    x = torch.randn(mask.shape, device=device, generator=generator)

    use_cfg = guidance_scale > 0.0
    if use_cfg:
        cond_full = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        mask_full = torch.cat([mask, mask], dim=0)
    else:
        cond_full, mask_full = cond, mask

    @torch.no_grad()
    def x0hat_eval(x_state, t_scalar):
        xb = torch.cat([x_state, x_state], dim=0) if use_cfg else x_state
        t = torch.full((xb.shape[0],), float(t_scalar), device=device)
        xhat = model(xb, mask_full, t, cond_full)
        if use_cfg:
            c, u = xhat[:B], xhat[B:]
            xhat = (1.0 + guidance_scale) * c - guidance_scale * u
        xhat = xhat * mask + (-1.0) * (1.0 - mask)          # padding -> void
        if pin is not None:
            xhat = xhat * (1.0 - pin) + 1.0 * pin           # pin cube -> solid
        return xhat

    def dxdt(x_state, t_scalar):
        t1 = torch.tensor([float(t_scalar)], device=device)
        mean = mean_fn(t1).view(1, 1, 1, 1, 1)
        std = std_fn(t1).view(1, 1, 1, 1, 1)
        x0hat = x0hat_eval(x_state, t_scalar)
        score = -(x_state - mean * x0hat) / (std ** 2)
        drift = drift_fn(t1).view(1, 1, 1, 1, 1)
        return drift * (x_state + score)

    s = torch.linspace(0.0, 1.0, n_steps + 1)
    ts = (eps + (1.0 - eps) * (1.0 - s) ** 2).tolist()
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        d0 = dxdt(x, t0)
        if i == n_steps - 1:
            x = x + dt * d0
        else:
            x_pred = x + dt * d0
            d1 = dxdt(x_pred, t1)
            x = x + 0.5 * dt * (d0 + d1)
    return x.cpu()


def pin_connectivity(binary, pin_true):
    """
    6-connectivity of solid voxels. Returns (in_largest, frac) where
    in_largest says whether any pinned voxel lies in the largest solid
    component, and frac = size of the largest component touching the pin /
    total solid. (None, None) if nothing is pinned or nothing is solid.
    """
    if pin_true is None or not pin_true.any():
        return None, None
    solid = binary > 0
    if not solid.any():
        return False, 0.0
    try:
        from scipy import ndimage
    except ImportError:
        return None, None
    lab, n = ndimage.label(solid)
    if n == 0:
        return False, 0.0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    pin_labels = np.unique(lab[pin_true & solid])
    pin_labels = pin_labels[pin_labels > 0]
    if pin_labels.size == 0:
        return False, 0.0
    frac = float(sizes[pin_labels].max()) / float(solid.sum())
    return bool(largest in set(pin_labels.tolist())), frac


def voxel_contact_fraction(binary_vox, points01, radius=1):
    """Fraction of points with >=1 solid voxel in their (2r+1)^3 cube."""
    if points01 is None:
        return None
    pts = np.atleast_2d(np.asarray(points01, dtype=np.float64))
    if pts.size == 0:
        return None
    Nx, Ny, Nz = binary_vox.shape
    hits = 0
    for p in pts:
        ix, iy, iz = point01_to_voxel_index(p, binary_vox.shape)
        if binary_vox[max(0, ix - radius):min(Nx, ix + radius + 1),
                      max(0, iy - radius):min(Ny, iy + radius + 1),
                      max(0, iz - radius):min(Nz, iz + radius + 1)].sum() > 0:
            hits += 1
    return hits / len(pts)


# --------------------------------------------------------------------------
# NEW: condition selection
# --------------------------------------------------------------------------

def reproduce_val_split(N_total, hp):
    """Byte-for-byte the split in make_loaders_masked, from hparams.json."""
    nsamples = hp.get("nsamples", 0)
    if nsamples is None or nsamples <= 0:
        nsamples = N_total
    nsamples = min(nsamples, N_total)
    rng = np.random.default_rng(hp["seed"])
    perm = rng.permutation(N_total)[:nsamples]
    n_val = max(1, int(hp["val_frac"] * nsamples))
    return perm[n_val:], perm[:n_val]


def indices_from_run(run_dir):
    """Parse cond##_idx#### folders of an earlier sampler run, in cond order."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--indices_from_run: not a directory: {run_dir}")
    j = run_dir / "chosen_indices.json"
    if j.exists():
        return [int(i) for i in json.load(open(j))["dataset_indices"]]
    found = []
    for d in run_dir.iterdir():
        m = re.fullmatch(r"cond(\d+)_idx(\d+)", d.name)
        if d.is_dir() and m:
            found.append((int(m.group(1)), int(m.group(2))))
    if not found:
        raise ValueError(f"No cond##_idx#### folders found in {run_dir}")
    return [idx for _, idx in sorted(found)]


def select_conditions(args, dataset, hp):
    N = len(dataset)
    val_set = None
    if args["val_only"]:
        tr, va = reproduce_val_split(N, hp)
        val_set = set(va.tolist())
        print(f"--val_only: split from hparams.json (seed={hp['seed']}, "
              f"nsamples={hp.get('nsamples', 0)}, val_frac={hp['val_frac']}) -> "
              f"{len(tr)} train / {len(va)} val")

    if args["indices_from_run"]:
        chosen = indices_from_run(args["indices_from_run"])
        mode = f"from_run:{args['indices_from_run']}"
        if args["indices"].strip():
            print("WARNING: --indices ignored because --indices_from_run is set")
    elif args["indices"].strip():
        chosen = [int(s) for s in args["indices"].split(",") if s.strip()]
        mode = "explicit"
    else:
        rng = np.random.default_rng(args["seed"])
        pool = np.asarray(sorted(val_set)) if val_set is not None else np.arange(N)
        n_pick = min(args["n_random"], len(pool))
        chosen = rng.choice(pool, size=n_pick, replace=False).tolist()
        mode = "random_val" if val_set is not None else "random"

    bad = [i for i in chosen if i < 0 or i >= N]
    if bad:
        raise ValueError(f"Indices out of range for this dataset (N={N}): {bad}. "
                         "Dataset indices are positions in the .npy, so they only "
                         "carry over between runs on the SAME data file.")
    if val_set is not None and mode != "random_val":
        dropped = [i for i in chosen if i not in val_set]
        if dropped:
            print(f"WARNING: --val_only dropping indices not in the val split: {dropped}")
        chosen = [i for i in chosen if i in val_set]
        if not chosen:
            raise ValueError("No requested indices fall in the val split.")
    return [int(i) for i in chosen], mode


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Route C ensemble sampler with load/BC solid pinning (v1001)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hparams", type=str, default=None)
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)

    # which conditions
    p.add_argument("--indices", type=str, default="")
    p.add_argument("--indices_from_run", type=str, default="",
                   help="Earlier sampler outdir; reuse its conditions in order.")
    p.add_argument("--n_random", default=4, type=int)
    p.add_argument("--val_only", action="store_true")
    p.add_argument("--noise_seed_mode", default="position",
                   choices=["position", "index"],
                   help="position: seed*1000+position (v0804 behavior). "
                        "index: seed*1000003+dataset_index.")

    # sampling
    p.add_argument("--ensemble_size", default=8, type=int)
    p.add_argument("--n_steps", default=250, type=int)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    p.add_argument("--guidance_scale", default=0.0, type=float)

    # pinning
    p.add_argument("--pin_load", action="store_true",
                   help="Force solid material at the load point every step.")
    p.add_argument("--pin_bc", action="store_true",
                   help="Force solid material at every BC point every step.")
    p.add_argument("--pin_radius", default=1, type=int,
                   help="Chebyshev radius of the pinned cube ((2r+1)^3 voxels).")
    p.add_argument("--contact_radius", default=1, type=int,
                   help="Radius for the load/BC contact metric.")

    # manufacturability override
    p.add_argument("--manuf_override", type=str, default="none",
                   choices=["none", "positive", "negative", "empty", "value"])
    p.add_argument("--manuf_override_percentile", type=float, default=None)
    p.add_argument("--manuf_override_raw_value", type=float, default=None)
    p.add_argument("--manufacturability_csv", type=str, default=None)
    return vars(p.parse_args())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    ckpt_path = Path(args["ckpt"])
    hparams_path = (Path(args["hparams"]) if args["hparams"]
                    else ckpt_path.parent.parent / "hparams.json")
    with open(hparams_path) as f:
        hp = json.load(f)
    print(f"Loaded hparams from {hparams_path}")

    meta_path = args["meta_path"] or hp.get("meta_path") or args["data_file"][:-4] + "_meta.npz"
    meta = np.load(meta_path, allow_pickle=True)
    dataset = NeuralFieldDataset3D(args["data_file"])

    model = MaskedScoreNet3D(
        cond_dim=dataset.cond_dim, cond_embed_dim=hp["cond_embed_dim"],
        cond_ch=hp["cond_ch"], c1=hp["unet_ch1_dim"], c2=hp["unet_ch2_dim"],
        c3=hp["unet_ch3_dim"], ed=hp["t_embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    if args["guidance_scale"] > 0.0 and hp.get("cond_drop_prob", 0.0) <= 0.0:
        print("WARNING: guidance_scale > 0 but cond_drop_prob = 0 in training; "
              "CFG is not valid for this checkpoint.")

    manuf_mode = str(meta["manufacturability_mode"]) if "manufacturability_mode" in meta else "omit"
    cond_slices = json.loads(str(meta["cond_slices_json"])) if "cond_slices_json" in meta else {}
    if manuf_mode != "omit":
        print(f"Dataset manufacturability_mode = {manuf_mode} "
              f"(threshold @ p{float(meta['manuf_percentile_threshold']):.0f} "
              f"= {float(meta['manuf_threshold_value']):.6g})")
    elif args["manuf_override"] != "none":
        print("WARNING: --manuf_override set but dataset has no manufacturability slot; ignored.")
    t_val, t_lbl = compute_manuf_override_targets(manuf_mode, meta, args)

    pin_any = args["pin_load"] or args["pin_bc"]
    R = args["pin_radius"]
    print(f"Pinning: load={'ON' if args['pin_load'] else 'off'}  "
          f"bc={'ON' if args['pin_bc'] else 'off'}  radius={R}")

    chosen, sel_mode = select_conditions(args, dataset, hp)
    print(f"Sampling conditions ({sel_mode}): {chosen}")

    outdir = Path(args["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "chosen_indices.json", "w") as f:
        json.dump({
            "dataset_indices": chosen,
            "source_indices": [_source_index(dataset.sample_infos[i]) for i in chosen],
            "selection_mode": sel_mode,
            "data_file": str(args["data_file"]),
            "ckpt": str(ckpt_path),
            "args": args,
        }, f, indent=2, default=str)

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)
    E = args["ensemble_size"]
    run_rows = []

    def _tolist(a):
        return None if a is None else np.asarray(a, dtype=float).tolist()

    for ci, idx in enumerate(chosen):
        vox = dataset.voxels[idx]
        Nx, Ny, Nz = vox.shape
        Px, Py, Pz = pad_to_multiple(Nx), pad_to_multiple(Ny), pad_to_multiple(Nz)
        mask = torch.zeros(E, 1, Px, Py, Pz)
        mask[:, :, :Nx, :Ny, :Nz] = 1.0

        cond_vec = dataset.conds[idx]
        original_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
        cond_vec = apply_manuf_override(cond_vec, cond_slices, manuf_mode,
                                        args["manuf_override"], t_val, t_lbl)
        applied_manuf = decode_manuf_from_cond(cond_vec, cond_slices, manuf_mode)
        cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)
        overlay = decode_condition_overlay(cond_vec, meta)

        # ---- pin mask ----
        pin_pad = np.zeros((Px, Py, Pz), dtype=bool)
        pin_centers = {}
        if args["pin_load"]:
            if overlay["load_point"] is None:
                print(f"  WARNING: idx {idx} has no decodable load point; load pin skipped")
            else:
                m, c = build_solid_pin_mask((Nx, Ny, Nz), (Px, Py, Pz),
                                            overlay["load_point"][None], R)
                pin_pad |= m
                pin_centers["load"] = c
        if args["pin_bc"]:
            if overlay["bc_points"] is None or len(overlay["bc_points"]) == 0:
                print(f"  WARNING: idx {idx} has no decodable BC points; BC pin skipped")
            else:
                m, c = build_solid_pin_mask((Nx, Ny, Nz), (Px, Py, Pz),
                                            overlay["bc_points"], R)
                pin_pad |= m
                pin_centers["bc"] = c
        pin_true = pin_pad[:Nx, :Ny, :Nz]
        solid_pin = None
        if pin_any and pin_pad.any():
            solid_pin = torch.from_numpy(pin_pad.astype(np.float32))[None, None].expand(E, 1, Px, Py, Pz)

        # ---- solve ----
        noise_seed = (args["seed"] * 1000 + ci if args["noise_seed_mode"] == "position"
                      else args["seed"] * 1000003 + idx)
        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed)
        x_final = heun_sampler_masked_pinned(
            model, mask, cond, mean_fn, std_fn, drift_fn,
            n_steps=args["n_steps"], eps=args["sample_eps"], device=device,
            guidance_scale=args["guidance_scale"], generator=gen,
            solid_pin=solid_pin,
        ).numpy()[:, 0, :Nx, :Ny, :Nz]

        # final enforcement, counted (loud flag if nonzero)
        n_forced = [0] * E
        if solid_pin is not None:
            for e in range(E):
                bad = pin_true & (x_final[e] <= 0.0)
                n_forced[e] = int(bad.sum())
                x_final[e][bad] = 1.0

        bins = [(x_final[e] > 0.0).astype(np.float32) for e in range(E)]
        masses = [float(b.mean()) for b in bins]
        speckles = [speckle_fraction(b) for b in bins]
        iou = pairwise_iou(bins)
        gt_mass = float(vox.mean())
        lp = overlay["load_point"][None] if overlay["load_point"] is not None else None
        load_c = [voxel_contact_fraction(b, lp, args["contact_radius"]) for b in bins]
        bc_c = [voxel_contact_fraction(b, overlay["bc_points"], args["contact_radius"]) for b in bins]
        conn = [pin_connectivity(b, pin_true if solid_pin is not None else None) for b in bins]

        # ---- plots ----
        cdir = outdir / f"cond{ci:02d}_idx{idx}"
        cdir.mkdir(exist_ok=True)
        if manuf_mode == "omit":
            msuf = ""
        elif not applied_manuf["has_label"]:
            msuf = " | manuf: UNLABELED"
        elif manuf_mode == "scalar":
            msuf = f" | manuf: value={applied_manuf['value']:.4f}"
        else:
            msuf = f" | manuf: {'POSITIVE' if applied_manuf['label'] > 0 else 'NEGATIVE'}"
        psuf = ""
        if solid_pin is not None:
            psuf = " | pin:" + "+".join(k for k in ("load", "bc") if k in pin_centers) + f" r={R}"
        for e in range(E):
            plot_voxel_with_overlays_implicit(
                bins[e], overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
                title=(f"idx {idx} member {e} | ({Nx},{Ny},{Nz}) | mass {masses[e]:.3f}"
                       f" | speckle {speckles[e]:.3f}{msuf}{psuf}"),
                save_path=cdir / f"member{e:02d}.png",
            )
        plot_voxel_with_overlays_implicit(
            vox, overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
            title=f"GT idx {idx} | ({Nx},{Ny},{Nz}) | mass {gt_mass:.3f}",
            save_path=cdir / "gt.png",
        )

        sample_info = dataset.sample_infos[idx]
        src = _source_index(sample_info)
        bc_dofs = parse_bc_dofs_from_cond_str(dataset.cond_strs[idx])
        np.savez_compressed(
            cdir / "ensemble.npz",
            fields=x_final, binaries=np.stack(bins), cond=cond_vec, gt=vox,
            dataset_index=idx, source_index=src,
            cond_str=str(dataset.cond_strs[idx]), bc_dofs=bc_dofs,
            guidance_scale=args["guidance_scale"],
            manuf_mode=manuf_mode, manuf_override=args["manuf_override"],
            pin_mask=pin_true, pin_load=args["pin_load"], pin_bc=args["pin_bc"],
            pin_radius=R,
        )

        manifest = {
            "provenance": {
                "dataset_index": idx, "source_index": src,
                "data_file": str(args["data_file"]), "meta_path": str(meta_path),
                "ckpt": str(ckpt_path), "sampler": "v1001_masked_voxel_diff_sampler.py",
                "generated": datetime.now().isoformat(timespec="seconds"),
            },
            "part": {
                "shape": [Nx, Ny, Nz], "gt_mass_fraction": gt_mass,
                "cond_str": str(dataset.cond_strs[idx]),
                "sample_info": ({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in dict(sample_info).items()}
                                if isinstance(sample_info, dict) else None),
            },
            "conditions_decoded": {
                "coordinate_convention": "[0,1] normalized by max(Nx,Ny,Nz); "
                                         "voxel index ~= coord * (max_dim - 1)",
                "bc_points": _tolist(overlay["bc_points"]),
                "bc_dofs": bc_dofs,
                "load_point": _tolist(overlay["load_point"]),
                "load_dir": _tolist(overlay["load_dir"]),
                "conditioning_spec": overlay["conditioning_spec"],
                "cond_vector": cond_vec.astype(float).tolist(),
            },
            "manufacturability": {
                "mode": manuf_mode, "override_requested": args["manuf_override"],
                "as_originally_drawn": original_manuf,
                "as_applied_to_this_sample": applied_manuf,
            },
            "pinning": {
                "pin_load": args["pin_load"], "pin_bc": args["pin_bc"],
                "pin_radius": R, "pin_center_voxels": pin_centers,
                "n_pinned_voxels": int(pin_true.sum()),
                "n_pin_voxels_forced_at_end": n_forced,
                "pin_in_largest_component": [c[0] for c in conn],
                "pin_component_frac": [None if c[1] is None else round(c[1], 4) for c in conn],
            },
            "sampling": {
                "ensemble_size": E, "n_steps": args["n_steps"],
                "sample_eps": args["sample_eps"], "guidance_scale": args["guidance_scale"],
                "seed": args["seed"], "noise_seed_mode": args["noise_seed_mode"],
                "noise_seed": noise_seed,
            },
            "results": {
                "member_mass_fractions": [round(m, 5) for m in masses],
                "member_speckle_fractions": [round(s, 5) for s in speckles],
                "pairwise_iou": round(iou, 5) if iou == iou else None,
                "contact_radius": args["contact_radius"],
                "member_load_contact": load_c,
                "member_bc_contact": bc_c,
            },
        }
        with open(cdir / "manifest.json", "w") as jf:
            json.dump(manifest, jf, indent=2, default=str)

        print(f"[cond {ci}] idx {idx} (source_index {src}) shape ({Nx},{Ny},{Nz})  GT mass {gt_mass:.3f}")
        print(f"          member masses  {[round(m, 3) for m in masses]}")
        print(f"          speckle fracs  {[round(s, 3) for s in speckles]}")
        print(f"          pairwise IoU   {iou:.3f}")
        print(f"          load contact   {load_c}   bc contact {[None if b is None else round(b, 3) for b in bc_c]}")
        if solid_pin is not None:
            flag = "  <-- pin not honored by trajectory" if any(n_forced) else ""
            print(f"          pinned voxels  {int(pin_true.sum())}  forced at end {n_forced}{flag}")
            print(f"          pin in largest component {[c[0] for c in conn]}")
        if manuf_mode != "omit":
            print(f"          manuf (applied) {applied_manuf}")

        run_rows.append({
            "dataset_index": idx, "source_index": src,
            "mean_load_contact": float(np.mean([c for c in load_c if c is not None])) if any(c is not None for c in load_c) else None,
            "mean_bc_contact": float(np.mean([c for c in bc_c if c is not None])) if any(c is not None for c in bc_c) else None,
            "n_forced_total": int(sum(n_forced)),
            "frac_pin_in_largest": (float(np.mean([bool(c[0]) for c in conn])) if conn and conn[0][0] is not None else None),
            "pairwise_iou": iou if iou == iou else None,
        })

    def _avg(key):
        v = [r[key] for r in run_rows if r[key] is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "n_conditions": len(run_rows), "ensemble_size": E,
        "pin_load": args["pin_load"], "pin_bc": args["pin_bc"], "pin_radius": R,
        "mean_load_contact": _avg("mean_load_contact"),
        "mean_bc_contact": _avg("mean_bc_contact"),
        "total_pin_voxels_forced_at_end": int(sum(r["n_forced_total"] for r in run_rows)),
        "mean_frac_pin_in_largest_component": _avg("frac_pin_in_largest"),
        "mean_pairwise_iou": _avg("pairwise_iou"),
        "per_condition": run_rows,
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nRun summary:", {k: v for k, v in summary.items() if k != "per_condition"})
    print(f"Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()
