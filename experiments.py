"""Comparative experiments for PCA vs ICA and noise sensitivity evaluation."""

import os
import matplotlib.pyplot as plt
import numpy as np
from ica import center, fastica, match_and_score, whiten
from run_ica import MIXING_MATRIX, SOURCE_FILES, load_sources, save_wav


def run_pca_vs_ica(S, X, names):
    """Compare PCA whitening baseline against FastICA independent source recovery."""
    X_centered, _ = center(X)
    Z, _ = whiten(X_centered)  # PCA components: uncorrelated rows
    S_ica, _ = fastica(Z)

    score_pca = match_and_score(S, Z)
    score_ica = match_and_score(S, S_ica)

    print(f"\n{'Source':<10} | {'PCA |corr|':<10} | {'PCA SDR':<9} | {'ICA |corr|':<10} | {'ICA SDR':<9}")
    print("-" * 58)
    for idx, name in enumerate(names):
        print(f"{name:<10} | {score_pca['abs_correlations'][idx]:<10.4f} | {score_pca['sdr_db'][idx]:<9.2f} | "
              f"{score_ica['abs_correlations'][idx]:<10.4f} | {score_ica['sdr_db'][idx]:<9.2f}")

    # Bar chart comparing SDR per source
    os.makedirs("results", exist_ok=True)
    x_indices = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x_indices - width / 2, score_pca["sdr_db"], width, label="PCA (whitening)")
    plt.bar(x_indices + width / 2, score_ica["sdr_db"], width, label="FastICA")
    plt.xticks(x_indices, names)
    plt.ylabel("SDR (dB)")
    plt.title("Separation Quality: PCA vs ICA")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pca_vs_ica.png", dpi=120)
    plt.close()


def run_snr_sweep(S, X, sample_rate, names):
    """Evaluate FastICA separation performance degradation across varying SNR noise levels."""
    snr_levels = [40, 30, 20, 10, 5, 0]
    mean_sdrs, mean_corrs = [], []
    np.random.seed(0)

    # Signal power per mixture row, shape: (3, 1)
    signal_power = np.mean(X**2, axis=1, keepdims=True)

    print(f"\n{'SNR (dB)':<10} | {'Mean SDR (dB)':<14} | {'Mean |corr|':<12}")
    print("-" * 42)
    for snr in snr_levels:
        noise_power = signal_power / (10 ** (snr / 10.0))
        # X_noisy shape: (3, n_samples)
        X_noisy = X + np.random.normal(0, np.sqrt(noise_power), X.shape)
        X_c, _ = center(X_noisy)
        Z, _ = whiten(X_c)
        S_est, _ = fastica(Z)
        res = match_and_score(S, S_est)

        m_sdr = float(np.mean(res["sdr_db"]))
        m_corr = float(np.mean(res["abs_correlations"]))
        mean_sdrs.append(m_sdr)
        mean_corrs.append(m_corr)
        print(f"{snr:<10} | {m_sdr:<14.2f} | {m_corr:<12.4f}")

        # Save audio sample for 10 dB noise level
        if snr == 10:
            for idx, name in enumerate(names):
                save_wav(f"separated/noisy10dB_{name}.wav", res["S_aligned"][idx], sample_rate)

    # Plot SNR sweep curve
    plt.figure(figsize=(6, 4))
    plt.plot(snr_levels, mean_sdrs, marker="o", color="tab:red", lw=1.5)
    plt.xlabel("Input Mixture SNR (dB)")
    plt.ylabel("Output Mean SDR (dB)")
    plt.title("FastICA Performance vs Input Noise Level")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig("results/snr_sweep.png", dpi=120)
    plt.close()


def main():
    """Execute comparative experiments for PCA vs ICA and noise robustness sweeps."""
    S, sample_rate = load_sources(SOURCE_FILES)
    # X shape: (3, n_samples)
    X = MIXING_MATRIX @ S
    source_names = [os.path.splitext(os.path.basename(f))[0] for f in SOURCE_FILES]

    run_pca_vs_ica(S, X, source_names)
    run_snr_sweep(S, X, sample_rate, source_names)

    # Experiment 3 (underdetermined system: 2 mixtures for 3 sources fails):
    # X_under = MIXING_MATRIX[:2] @ S  # shape: (2, n_samples)
    # Z_under, _ = whiten(center(X_under)[0])  # can only yield 2 outputs for 3 sources


if __name__ == "__main__":
    main()
