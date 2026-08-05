"""
Speed patch for v0720_field_diff_perceiver_trainer.py.

Two replacements:

  1. heun_sampler_field_batched -- replaces ode_sampler_field.
     * Fixed-step Heun (2nd order) on the probability-flow ODE: n_steps=100
       Heun steps ~= 200 NFE, vs. RK45's adaptive 500-1500+ NFE at 1e-4 tol.
       Fixed steps are fine here (context doc already flagged this); Heun at
       100-150 steps is visually indistinguishable from RK45 for binarized
       occupancy fields.
     * Whole ensemble solved as ONE batch (E in the batch dim): E members
       share every kernel launch instead of E sequential scipy solves.
     * Stays on GPU end-to-end (no scipy, no float64, no CPU round trips).
     * Optional fp16 autocast for the network evals (state kept fp32).
     * Optional classifier-free guidance (uncond = zero cond vector, valid
       because of training cond dropout), done in the same batch by
       doubling B.

     Expected speedup vs. current path: ~1 order of magnitude or more,
     depending on part size.

  2. validate_fast -- replaces validate.
     * Batches val parts through the same DataLoader machinery as training
       (B=16 instead of B=1) and optionally caps the number of val parts.
     * Uses a fixed seed for t/noise/point draws so the val metric is a
       low-variance, comparable-across-epochs number (the current val loss
       is stochastic in t, which makes "best checkpoint" selection noisy).

Usage in the trainer:

    from v0720_fast_patch import heun_sampler_field_batched, validate_fast

    # in save_ensembles(): replace the per-member loop with one call
    y_all = heun_sampler_field_batched(
        model, coords, cond, mean_fn, std_fn, drift_fn,
        ensemble_size=config["ensemble_size"],
        n_steps=120,
        n_context=config["sample_context"],
        eval_chunk=config["eval_chunk"],
        device=device, base_seed=1000 * int(idx),
    )                                # [E, N_pts, 1]
    for e in range(config["ensemble_size"]):
        x_bin = (y_all[e].cpu().numpy().reshape(Nx, Ny, Nz) > 0.0)
        ...

    # in main(): replace validate(...) with
    val_loss = validate_fast(
        model, dataset, val_indices, mean_fn, std_fn, device,
        eps, config["loss_weighting"], config["n_points"],
        config["n_context"], batchsize=16, max_parts=128,
    )

Requires v0720_neural_diff_trainer_fixed.py importable (same dir), for
PointSampleDataset only in validate_fast.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from v0720_neural_diff_trainer_fixed import PointSampleDataset
from v0720_field_diff_perceiver_trainer import field_diffusion_loss


# ---------------------------------------------------------------------------
# 1. Batched fixed-step Heun sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def heun_sampler_field_batched(
    model, coords, cond, mean_fn, std_fn, drift_fn,
    ensemble_size=4, n_steps=120, n_context=4096, eval_chunk=16384,
    eps=1e-3, device="cuda", base_seed=0,
    guidance_scale=0.0, use_amp=True,
):
    """
    Sample `ensemble_size` members of the field in ONE batched trajectory.

    coords: [N_pts, 3] full-grid coordinates (cpu or gpu)
    cond:   [cond_dim]
    Returns y_final: [E, N_pts, 1] (float32, on CPU).

    Semantics match ode_sampler_field: per-member fixed context indices
    (drawn from a member-specific generator so results are reproducible),
    latents re-encoded from current noisy context values at every network
    evaluation, X0-prediction score, probability-flow ODE from t=1 to eps.
    """
    model.eval()
    E = int(ensemble_size)
    N = coords.shape[0]
    coords_dev = coords.to(device)
    coords_b = coords_dev.unsqueeze(0).expand(E, -1, -1).contiguous()  # [E,N,3]
    cond_b = cond.to(device).unsqueeze(0).expand(E, -1).contiguous()   # [E,C]

    use_cfg = guidance_scale != 0.0
    if use_cfg:
        # batch = [cond members ; uncond members]
        cond_full = torch.cat([cond_b, torch.zeros_like(cond_b)], dim=0)
        coords_full = torch.cat([coords_b, coords_b], dim=0)
        B = 2 * E
    else:
        cond_full = cond_b
        coords_full = coords_b
        B = E

    gens = [torch.Generator().manual_seed(int(base_seed) + e) for e in range(E)]
    n_ctx = min(n_context, N)
    ctx_idx = torch.stack(
        [torch.randperm(N, generator=g)[:n_ctx] for g in gens]
    ).to(device)                                                       # [E,Mc]
    if use_cfg:
        ctx_idx_full = torch.cat([ctx_idx, ctx_idx], dim=0)            # [2E,Mc]
    else:
        ctx_idx_full = ctx_idx
    coords_c = torch.gather(
        coords_full, 1, ctx_idx_full.unsqueeze(-1).expand(-1, -1, 3)
    )                                                                  # [B,Mc,3]

    y = torch.stack(
        [torch.randn(N, 1, generator=g) for g in gens]
    ).to(device)                                                       # [E,N,1]

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16,
                             enabled=(use_amp and str(device).startswith("cuda")))

    def y0hat_eval(y_state, t_scalar):
        """y_state: [E,N,1] fp32 -> y0hat (guided if CFG): [E,N,1] fp32."""
        y_full = torch.cat([y_state, y_state], dim=0) if use_cfg else y_state
        t = torch.full((B,), float(t_scalar), device=device)
        y_c = torch.gather(y_full, 1, ctx_idx_full.unsqueeze(-1))      # [B,Mc,1]
        with amp_ctx:
            lat = model.compute_latents(coords_c, y_c, t, cond_full)
            outs = []
            for s in range(0, N, eval_chunk):
                e_ = min(s + eval_chunk, N)
                outs.append(model.decode_queries(
                    lat, coords_full[:, s:e_], y_full[:, s:e_], t, cond_full
                ))
        y0hat = torch.cat(outs, dim=1).float()                         # [B,N,1]
        if use_cfg:
            c, u = y0hat[:E], y0hat[E:]
            y0hat = (1.0 + guidance_scale) * c - guidance_scale * u
        return y0hat

    def dydt(y_state, t_scalar):
        t1 = torch.tensor([float(t_scalar)], device=device)
        mean = mean_fn(t1).view(1, 1, 1)
        std = std_fn(t1).view(1, 1, 1)
        y0hat = y0hat_eval(y_state, t_scalar)
        score = -(y_state - mean * y0hat) / (std ** 2)
        drift = drift_fn(t1).view(1, 1, 1)
        return drift * (y_state + score)

    # Heun (explicit trapezoid), t: 1 -> eps. Quadratic spacing puts more
    # steps near t=0 where the field sharpens; swap for linspace if preferred.
    s = torch.linspace(0.0, 1.0, n_steps + 1)
    ts = (eps + (1.0 - eps) * (1.0 - s) ** 2).tolist()   # ts[0]=1.0, ts[-1]=eps
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0                                     # negative
        d0 = dydt(y, t0)
        if i == n_steps - 1:
            y = y + dt * d0                              # final Euler step
        else:
            y_pred = y + dt * d0
            d1 = dydt(y_pred, t1)
            y = y + 0.5 * dt * (d0 + d1)

    return y.cpu()


# ---------------------------------------------------------------------------
# 2. Batched, deterministic validation
# ---------------------------------------------------------------------------

def validate_fast(model, dataset, val_indices, mean_fn, std_fn, device,
                  eps, loss_weighting, n_points, n_context,
                  batchsize=16, max_parts=128, seed=1234):
    """
    Batched replacement for validate(). Also fixes the torch RNG seed for
    the duration of the call so t / noise / point draws are identical every
    epoch -> val curve is smooth and 'best checkpoint' selection is stable.
    """
    idxs = np.asarray(val_indices)
    if max_parts is not None and len(idxs) > max_parts:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(idxs, size=max_parts, replace=False)

    ds = PointSampleDataset(dataset, idxs, n_points)
    loader = DataLoader(ds, batch_size=batchsize, shuffle=False,
                        num_workers=0, pin_memory=True)

    model.eval()
    total, n = 0.0, 0
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            for coords_b, y0_b, cond_b, _ in loader:
                loss = field_diffusion_loss(
                    model,
                    y0_b.to(device, non_blocking=True),
                    coords_b.to(device, non_blocking=True),
                    cond_b.to(device, non_blocking=True),
                    n_context, mean_fn, std_fn,
                    eps=eps, loss_weighting=loss_weighting,
                    cond_drop_prob=0.0,
                )
                total += loss.item()
                n += 1
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    model.train()
    return total / max(1, n)