import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Plot GAN training metrics from checkpoint directories.')
    p.add_argument('checkpoint_dirs', nargs='+', help='One or more checkpoint directories containing metrics.csv')
    p.add_argument('--rolling', type=int, default=0, help='Optional rolling window for smoothing curves')
    p.add_argument('--overlay-name', type=str, default='training_metrics_overlay.png', help='Filename for overlay plot when multiple runs are given')
    return p.parse_args()


def parse_steps_per_epoch(dirname):
    m_bs = re.search(r"bs(\d+)", dirname)
    m_ns = re.search(r"nsamp(\d+)", dirname)
    if m_bs and m_ns:
        batch_size = int(m_bs.group(1))
        n_samples = int(m_ns.group(1))
        steps_per_epoch = max(1, n_samples // batch_size)
        return steps_per_epoch, batch_size, n_samples
    return None, None, None


def load_metrics(checkpoint_dir):
    metrics_path = os.path.join(checkpoint_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f'No metrics.csv found at {metrics_path}')

    dirname = os.path.basename(os.path.normpath(checkpoint_dir))
    steps_per_epoch, batch_size, n_samples = parse_steps_per_epoch(dirname)

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

    # Base metrics (compatible with old trainer)
    base_cols = [
        'L_D_real', 'L_D_neg', 'L_D_fake',
        'L_G', 'D_grad_norm', 'G_grad_norm', 'L_div'
    ]
    # New probability metrics from the non‑UNet trainer
    prob_cols = [
        'P_pos_real_pos',  # P(class=1 | real positive)
        'P_neg_real_neg',  # P(class=2 | real negative)
        'P_fake_fake_D',   # P(class=0 | fake, D)
        'P_pos_fake_D',    # P(class=1 | fake, D)
        'P_pos_fake_G',    # P(class=1 | fake, G)
    ]

    for col in base_cols + prob_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    meta = {
        'run_name': dirname,
        'steps_per_epoch': steps_per_epoch,
        'batch_size': batch_size,
        'n_samples': n_samples,
        'metrics_path': metrics_path,
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
    if meta['batch_size'] is not None and meta['n_samples'] is not None:
        title_suffix += f" | bs={meta['batch_size']} nsamp={meta['n_samples']}"

    x_step = df['step']
    x_epoch = df['epoch']

    # ---- Figure 1: losses + grad norms (as before) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'GAN Training Metrics: {title_suffix}', fontsize=14)

    # D losses vs step
    ax = axes[0, 0]
    for col, label in [('L_D_real', 'D_real'), ('L_D_neg', 'D_neg'), ('L_D_fake', 'D_fake')]:
        if not df[col].isna().all():
            ax.plot(x_step, maybe_smooth(df[col], rolling), label=label, alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator losses vs step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # G loss vs step
    ax = axes[0, 1]
    if not df['L_G'].isna().all():
        ax.plot(x_step, maybe_smooth(df['L_G'], rolling), color='green', linewidth=2, label='L_G')
    ax.set_xlabel('Step')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss vs step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Grad norms vs epoch
    ax = axes[1, 0]
    for col, label in [('D_grad_norm', 'D_grad_norm'), ('G_grad_norm', 'G_grad_norm')]:
        if not df[col].isna().all():
            ax.plot(x_epoch, maybe_smooth(df[col], rolling), label=label, alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient norm')
    ax.set_title('Gradient norms vs epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # G loss + diversity vs epoch
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

    # ---- Figure 2: detailed losses (as before) ----
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig2.suptitle(f'GAN Loss Detail: {title_suffix}', fontsize=14)

    # All losses vs epoch
    ax = axes2[0]
    for col, label in [('L_D_real', 'D_real'), ('L_D_neg', 'D_neg'),
                       ('L_D_fake', 'D_fake'), ('L_G', 'L_G')]:
        if not df[col].isna().all():
            ax.plot(x_epoch, maybe_smooth(df[col], rolling), label=label, alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('All losses vs epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # G loss with smoothing vs step
    ax = axes2[1]
    if not df['L_G'].isna().all():
        ax.plot(x_step, df['L_G'], color='green', alpha=0.25, label='L_G raw')
        ax.plot(
            x_step,
            maybe_smooth(df['L_G'], max(rolling, 25) if rolling == 0 else rolling),
            color='black',
            linewidth=2,
            label='L_G smooth'
        )
    ax.set_xlabel('Step')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss with smoothing')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path2 = os.path.join(checkpoint_dir, 'training_metrics_detailed.png')
    save_plot(fig2, out_path2)

    # # ---- Figure 3: probability diagnostics ----
    # # Only create if at least one probability column is non‑NaN
    # prob_cols = [
    #     ('P_pos_real_pos', 'P(class=1 | real pos)'),
    #     ('P_neg_real_neg', 'P(class=2 | real neg)'),
    #     ('P_fake_fake_D',  'P(class=0 | fake, D)'),
    #     ('P_pos_fake_D',   'P(class=1 | fake, D)'),
    #     ('P_pos_fake_G',   'P(class=1 | fake, G)'),
    # ]
    # has_any_prob = any(col in df.columns and not df[col].isna().all()
    #                    for col, _ in prob_cols)
    # if has_any_prob:
    #     fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    #     fig3.suptitle(f'GAN Probability Diagnostics: {title_suffix}', fontsize=14)

    #     # Panel 1: discriminator probabilities vs step (real/fake)
    #     ax = axes3[0]
    #     for col, label in [
    #         ('P_pos_real_pos', 'P(class=1 | real pos)'),
    #         ('P_neg_real_neg', 'P(class=2 | real neg)'),
    #         ('P_fake_fake_D',  'P(class=0 | fake, D)'),
    #         ('P_pos_fake_D',   'P(class=1 | fake, D)'),
    #     ]:
    #         if col in df.columns and not df[col].isna().all():
    #             ax.plot(
    #                 x_step,
    #                 maybe_smooth(df[col], rolling),
    #                 label=label,
    #                 alpha=0.85
    #             )
    #     ax.set_xlabel('Step')
    #     ax.set_ylabel('Probability')
    #     ax.set_ylim(0.0, 1.0)
    #     ax.set_title('Discriminator probabilities vs step')
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(fontsize=8)

    #     # Panel 2: generator-target probability vs step
    #     ax = axes3[1]
    #     col = 'P_pos_fake_G'
    #     if col in df.columns and not df[col].isna().all():
    #         ax.plot(
    #             x_step,
    #             df[col],
    #             color='blue',
    #             alpha=0.25,
    #             label='P(class=1 | fake, G) raw'
    #         )
    #         ax.plot(
    #             x_step,
    #             maybe_smooth(df[col], max(rolling, 25) if rolling == 0 else rolling),
    #             color='black',
    #             linewidth=2,
    #             label='P(class=1 | fake, G) smooth'
    #         )
    #     ax.set_xlabel('Step')
    #     ax.set_ylabel('Probability')
    #     ax.set_ylim(0.0, 1.0)
    #     ax.set_title('Generator success probability vs step')
    #     ax.grid(True, alpha=0.3)
    #     ax.legend()

    #     out_path3 = os.path.join(checkpoint_dir, 'training_metrics_probs.png')
    #     save_plot(fig3, out_path3)

    # ---- Figure 3: probability diagnostics ----
    # Less busy version: split into real/fake/G panels and use stronger smoothing.
    prob_cols = [
        ('P_pos_real_pos', 'P(class=1 | real pos)'),
        ('P_neg_real_neg', 'P(class=2 | real neg)'),
        ('P_fake_fake_D',  'P(class=0 | fake, D)'),
        ('P_pos_fake_D',   'P(class=1 | fake, D)'),
        ('P_pos_fake_G',   'P(class=1 | fake, G)'),
    ]
    has_any_prob = any(col in df.columns and not df[col].isna().all()
                       for col, _ in prob_cols)

    if has_any_prob:
        prob_rolling = rolling if rolling and rolling > 1 else 50

        fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig3.suptitle(f'GAN Probability Diagnostics: {title_suffix}', fontsize=14)

        # Panel 1: real-sample classification probabilities
        ax = axes3[0]
        for col, label, color in [
            ('P_pos_real_pos', 'P(class=1 | real pos)', 'tab:green'),
            ('P_neg_real_neg', 'P(class=2 | real neg)', 'tab:orange'),
        ]:
            if col in df.columns and not df[col].isna().all():
                ax.plot(
                    x_step,
                    maybe_smooth(df[col], prob_rolling),
                    label=label,
                    color=color,
                    linewidth=2.2,
                )
        ax.set_ylabel('Probability')
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Real-sample classification')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

        # Panel 2: fake-sample discriminator probabilities
        ax = axes3[1]
        for col, label, color in [
            ('P_fake_fake_D', 'P(class=0 | fake, D)', 'tab:red'),
            ('P_pos_fake_D',  'P(class=1 | fake, D)', 'tab:purple'),
        ]:
            if col in df.columns and not df[col].isna().all():
                ax.plot(
                    x_step,
                    maybe_smooth(df[col], prob_rolling),
                    label=label,
                    color=color,
                    linewidth=2.2,
                )
        ax.set_ylabel('Probability')
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Fake-sample discriminator outputs')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

        # Panel 3: generator success probability
        ax = axes3[2]
        col = 'P_pos_fake_G'
        if col in df.columns and not df[col].isna().all():
            ax.plot(
                x_step,
                df[col],
                color='tab:blue',
                alpha=0.12,
                linewidth=1.0,
                label='raw'
            )
            ax.plot(
                x_step,
                maybe_smooth(df[col], prob_rolling),
                color='black',
                linewidth=2.5,
                label=f'smoothed ({prob_rolling})'
            )
        ax.set_xlabel('Step')
        ax.set_ylabel('Probability')
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Generator success probability')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

        fig3.tight_layout(rect=[0, 0, 0.84, 0.96])
        out_path3 = os.path.join(checkpoint_dir, 'training_metrics_probs.png')
        fig3.savefig(out_path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)


def plot_overlay(runs, overlay_name='training_metrics_overlay.png', rolling=0):
    if len(runs) < 2:
        return None

    parent_dirs = {os.path.dirname(os.path.normpath(ckpt)) for _, ckpt, _ in runs}
    overlay_dir = parent_dirs.pop() if len(parent_dirs) == 1 else os.getcwd()
    overlay_path = os.path.join(overlay_dir, overlay_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    for df, ckpt, meta in runs:
        ax.plot(df['epoch'], maybe_smooth(df['L_G'], rolling), linewidth=1.8, label=meta['run_name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generator loss')
    ax.set_title('Generator loss overlay')
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