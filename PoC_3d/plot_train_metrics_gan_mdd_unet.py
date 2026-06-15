import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Plot UNet-GAN training metrics from checkpoint directories.')
    p.add_argument('checkpoint_dirs', nargs='+', help='One or more checkpoint directories containing metrics.csv')
    p.add_argument('--rolling', type=int, default=0, help='Optional rolling window for smoothing curves')
    p.add_argument('--overlay-name', type=str, default='training_metrics_overlay.png', help='Filename for overlay plot when multiple runs are given')
    return p.parse_args()


def parse_steps_per_epoch(dirname):
    m_bs = re.search(r"bs(\d+)", dirname)
    m_epochs = re.search(r"epochs(\d+)", dirname)
    m_ns = re.search(r"nsamp(\d+)", dirname)
    if m_bs and m_ns and m_epochs:
        batch_size = int(m_bs.group(1))
        n_samples = int(m_ns.group(1))
        num_epochs = int(m_epochs.group(1))
        pos_est = max(1, n_samples // 2)
        steps_per_epoch = max(1, pos_est // batch_size)
        return steps_per_epoch, batch_size, n_samples, num_epochs
    if m_bs and m_ns:
        batch_size = int(m_bs.group(1))
        n_samples = int(m_ns.group(1))
        pos_est = max(1, n_samples // 2)
        steps_per_epoch = max(1, pos_est // batch_size)
        return steps_per_epoch, batch_size, n_samples, None
    return None, None, None, None


def parse_hparams_txt(checkpoint_dir):
    hparams = {}
    hp_path = os.path.join(checkpoint_dir, 'hparams.txt')
    if not os.path.exists(hp_path):
        return hparams
    with open(hp_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                hparams[k.strip()] = v.strip()
    return hparams


def load_metrics(checkpoint_dir):
    metrics_path = os.path.join(checkpoint_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f'No metrics.csv found at {metrics_path}')

    dirname = os.path.basename(os.path.normpath(checkpoint_dir))
    steps_per_epoch, batch_size, n_samples, num_epochs = parse_steps_per_epoch(dirname)
    hparams = parse_hparams_txt(checkpoint_dir)

    df = pd.read_csv(metrics_path)
    if 'step' not in df.columns:
        raise ValueError('metrics.csv must contain a step column')

    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    if 'epoch' in df.columns:
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    elif steps_per_epoch is not None:
        df['epoch'] = df['step'] / steps_per_epoch
    else:
        df['epoch'] = np.arange(len(df), dtype=float)

    expected_cols = [
        'L_D_real', 'L_D_neg', 'L_D_fake',
        'L_G', 'D_grad_norm', 'G_grad_norm', 'L_div'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    meta = {
        'run_name': dirname,
        'steps_per_epoch': steps_per_epoch,
        'batch_size': batch_size,
        'n_samples': n_samples,
        'num_epochs': num_epochs,
        'metrics_path': metrics_path,
        'hparams': hparams,
    }
    return df, meta


def maybe_smooth(series, rolling):
    if rolling and rolling > 1:
        return series.rolling(window=rolling, min_periods=1).mean()
    return series


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_single_run(df, checkpoint_dir, meta, rolling=0):
    run_name = meta['run_name']
    title_suffix = run_name
    if meta['batch_size'] is not None:
        title_suffix += f" | bs={meta['batch_size']}"
    if meta['steps_per_epoch'] is not None:
        title_suffix += f" | steps/epoch≈{meta['steps_per_epoch']}"

    x_step = df['step']
    x_epoch = df['epoch']

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'UNet GAN Training Metrics: {title_suffix}', fontsize=14)

    ax = axes[0, 0]
    for col, label in [('L_D_real', 'D_real'), ('L_D_neg', 'D_neg'), ('L_D_fake', 'D_fake')]:
        if not df[col].isna().all():
            ax.plot(x_step, maybe_smooth(df[col], rolling), label=label, alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator losses vs step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if not df['L_G'].isna().all():
        ax.plot(x_step, maybe_smooth(df['L_G'], rolling), color='green', linewidth=2, label='L_G')
    ax.set_xlabel('Step')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss vs step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for col, label in [('D_grad_norm', 'D_grad_norm'), ('G_grad_norm', 'G_grad_norm')]:
        if not df[col].isna().all():
            ax.plot(x_epoch, maybe_smooth(df[col], rolling), label=label, alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient norm')
    ax.set_title('Gradient norms vs epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if not df['L_G'].isna().all():
        ax.plot(x_epoch, maybe_smooth(df['L_G'], rolling), color='green', linewidth=2, label='L_G')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss / diversity vs epoch')
    ax.grid(True, alpha=0.3)
    if not df['L_div'].isna().all():
        ax2 = ax.twinx()
        ax2.plot(x_epoch, maybe_smooth(df['L_div'], rolling), color='red', linestyle='--', alpha=0.7, label='L_div')
        ax2.set_ylabel('Diversity loss')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        ax.legend()

    out_path = os.path.join(checkpoint_dir, 'training_metrics.png')
    save_plot(fig, out_path)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig2.suptitle(f'UNet GAN Loss Detail: {title_suffix}', fontsize=14)

    ax = axes2[0]
    for col, label in [('L_D_real', 'D_real'), ('L_D_neg', 'D_neg'), ('L_D_fake', 'D_fake'), ('L_G', 'L_G')]:
        if not df[col].isna().all():
            ax.plot(x_epoch, maybe_smooth(df[col], rolling), label=label, alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('All losses vs epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes2[1]
    if not df['L_G'].isna().all():
        smooth_window = max(rolling, 25) if rolling == 0 else rolling
        ax.plot(x_step, df['L_G'], color='green', alpha=0.25, label='L_G raw')
        ax.plot(x_step, maybe_smooth(df['L_G'], smooth_window), color='black', linewidth=2, label=f'L_G smooth ({smooth_window})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss with smoothing')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path2 = os.path.join(checkpoint_dir, 'training_metrics_detailed.png')
    save_plot(fig2, out_path2)

    summary_path = os.path.join(checkpoint_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"run_name: {run_name}\n")
        f.write(f"metrics_path: {meta['metrics_path']}\n")
        f.write(f"steps_per_epoch_estimate: {meta['steps_per_epoch']}\n")
        f.write(f"batch_size: {meta['batch_size']}\n")
        f.write(f"n_samples: {meta['n_samples']}\n")
        f.write(f"num_epochs: {meta['num_epochs']}\n")
        if not df['L_G'].isna().all():
            best_idx = df['L_G'].idxmin()
            f.write(f"best_L_G: {df.loc[best_idx, 'L_G']} at step {df.loc[best_idx, 'step']} epoch {df.loc[best_idx, 'epoch']}\n")
            f.write(f"final_L_G: {df['L_G'].dropna().iloc[-1]}\n")
        if not df['L_D_real'].isna().all():
            f.write(f"final_L_D_real: {df['L_D_real'].dropna().iloc[-1]}\n")
        if not df['L_D_neg'].isna().all():
            f.write(f"final_L_D_neg: {df['L_D_neg'].dropna().iloc[-1]}\n")
        if not df['L_D_fake'].isna().all():
            f.write(f"final_L_D_fake: {df['L_D_fake'].dropna().iloc[-1]}\n")
        if not df['L_div'].isna().all():
            f.write(f"final_L_div: {df['L_div'].dropna().iloc[-1]}\n")
        if meta['hparams']:
            f.write('\n[hparams.txt]\n')
            for k, v in meta['hparams'].items():
                f.write(f"{k} = {v}\n")


def plot_overlay(runs, overlay_name='training_metrics_overlay.png', rolling=0):
    if len(runs) < 2:
        return None

    parent_dirs = {os.path.dirname(os.path.normpath(ckpt)) for _, ckpt, _ in runs}
    overlay_dir = parent_dirs.pop() if len(parent_dirs) == 1 else os.getcwd()
    overlay_path = os.path.join(overlay_dir, overlay_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    for df, ckpt, meta in runs:
        if not df['L_G'].isna().all():
            ax.plot(df['epoch'], maybe_smooth(df['L_G'], rolling), linewidth=1.8, label=meta['run_name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generator loss')
    ax.set_title('UNet generator loss overlay')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, overlay_path)
    return overlay_path


def main():
    args = parse_args()
    checkpoint_dirs = [d for d in args.checkpoint_dirs if os.path.isdir(d)]
    if not checkpoint_dirs:
        raise SystemExit('No valid checkpoint directories provided.')

    runs = []
    for ckpt in checkpoint_dirs:
        try:
            df, meta = load_metrics(ckpt)
            plot_single_run(df, ckpt, meta, rolling=args.rolling)
            runs.append((df, ckpt, meta))
            print(f"[{meta['run_name']}] wrote plots into {ckpt}")
        except Exception as e:
            print(f'Error processing {ckpt}: {e}')

    if len(runs) >= 2:
        overlay_path = plot_overlay(runs, overlay_name=args.overlay_name, rolling=args.rolling)
        if overlay_path is not None:
            print(f'Wrote overlay plot to {overlay_path}')


if __name__ == '__main__':
    main()