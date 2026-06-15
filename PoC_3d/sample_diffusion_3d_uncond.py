import argparse
from pathlib import Path
import functools
import numpy as np
import torch
import yaml
from scipy import integrate
import matplotlib.pyplot as plt

from diffusion_3d_uncond import (
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    ScoreNet3DVoxel,
    plot_voxel_grid_3d,
)


def ode_sampler_voxel_uncond(score_model, x_shape, marginal_prob_mean, marginal_prob_std, drift_coeff,
                             mode, batch_size=1, atol=1e-4, rtol=1e-4, device="cuda", eps=1e-3):
    t0 = 1.0
    init_x = torch.randn(x_shape, device=device)

    def score_eval_wrapper(sample_flat, time_steps):
        sample = torch.tensor(sample_flat, device=device, dtype=torch.float32).reshape(x_shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps, cond=None)
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t_scalar, x_flat):
        time_steps = np.ones((x_shape[0],), dtype=np.float32) * t_scalar
        t_torch = torch.tensor(t_scalar, device=device, dtype=torch.float32)
        drift = drift_coeff(t_torch).cpu().numpy()
        std = marginal_prob_std(t_torch).cpu().numpy()
        mean_scale = marginal_prob_mean(t_torch).cpu().numpy()

        if mode == "X0":
            x0hat_flat = score_eval_wrapper(x_flat, time_steps)
            score = -(x_flat - mean_scale * x0hat_flat) / (std ** 2)
        else:
            raise ValueError("Only X0 mode implemented in voxel sampler")

        return drift * (x_flat + score)

    res = integrate.solve_ivp(
        ode_func,
        (t0, eps),
        init_x.cpu().numpy().reshape(-1),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    nsamples = res.y.shape[1]
    x_traj = torch.tensor(res.y, device=device, dtype=torch.float32)
    x_traj = x_traj.view(*x_shape, nsamples)
    x_final = x_traj[..., -1]
    return x_traj, x_final


def parse_args():
    p = argparse.ArgumentParser(description="Sample 3D voxel-space VPSDE diffusion model")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_root", type=str, required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log_root = Path(args.log_root)
    model_dir = log_root / f"version_{args.version}"
    ckpt_file = model_dir / "checkpoints" / "ckpt_best.pth"
    params_file = model_dir / "hparams.yml"

    assert ckpt_file.exists(), f"Checkpoint not found: {ckpt_file}"
    assert params_file.exists(), f"hparams.yml not found: {params_file}"

    config = yaml.safe_load(params_file.read_text())
    img_size = config["img_size"]
    assert img_size == args.img_size, "img_size mismatch between config and CLI"

    samples_dir = model_dir / "samples_manual"
    samples_dir.mkdir(parents=True, exist_ok=True)

    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_coeff_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    score_model = ScoreNet3DVoxel(
        in_ch=1,
        cond_ch=0,
        c1=config["unet_ch1_dim"],
        c2=config["unet_ch2_dim"],
        c3=config["unet_ch3_dim"],
        ed=config["t_embed_dim"],
        mode=config["mode"],
        marginal_prob_std=marginal_prob_std_fn,
    ).to(device)

    state_dict = torch.load(ckpt_file, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()

    batch_size = args.num_samples
    x_shape = torch.Size([batch_size, 1, img_size, img_size, img_size])

    x_traj, x_final = ode_sampler_voxel_uncond(
        score_model,
        x_shape,
        marginal_prob_mean_fn,
        marginal_prob_std_fn,
        drift_coeff_fn,
        mode=config["mode"],
        batch_size=batch_size,
        atol=args.atol,
        rtol=args.rtol,
        device=device,
        eps=args.eps,
    )

    x_samples = x_final.cpu().numpy()
    x_bin = (x_samples > 0.0).astype(np.float32)
    np.save(samples_dir / "traj.npy", x_traj.cpu().numpy())
    np.save(samples_dir / "fields_continuous.npy", x_samples)
    np.save(samples_dir / "fields_binary.npy", x_bin)

    for i in range(min(batch_size, 8)):
        plot_voxel_grid_3d(
            x_bin[i, 0],
            title=f"Voxel DM sample {i}",
            save_path=samples_dir / f"sample_{i:02d}.png",
        )

    print(f"Saved voxel-space trajectories and samples under {samples_dir}")


if __name__ == "__main__":
    main()