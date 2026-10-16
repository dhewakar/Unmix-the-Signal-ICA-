"""End-to-end 3-source audio separation pipeline using scratch-built FastICA."""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from ica import center, fastica, kurtosis, match_and_score, whiten

SOURCE_FILES = ["sources/speech.wav", "sources/music.wav", "sources/ambient.wav"]
MIXING_MATRIX = np.array([[1.0, 0.6, 0.4], [0.5, 1.0, 0.3], [0.3, 0.7, 1.0]])


def load_sources(files):
    """Load, convert to mono float, synchronize lengths, and validate sampling rates."""
    signals, sample_rates = [], []
    for file_path in files:
        rate, data = wavfile.read(file_path)
        sample_rates.append(rate)
        # Average stereo channels to create a single mono channel
        mono = data.mean(axis=1) if data.ndim > 1 else data
        # Scale integer PCM to float range [-1.0, 1.0]
        float_sig = mono / 32768.0 if np.issubdtype(mono.dtype, np.integer) else mono
        signals.append(float_sig.astype(np.float64))

    if len(set(sample_rates)) > 1:
        raise ValueError("All input audio files must have identical sample rates.")

    # Trim all signals to the shortest sample count to maintain equal lengths
    min_samples = min(len(sig) for sig in signals)
    # S shape: (3, n_samples)
    S = np.array([sig[:min_samples] for sig in signals])
    return S, sample_rates[0]


def save_wav(path, signal, sample_rate):
    """Normalize signal headroom, cast to int16, and write out to audio file."""
    # Ensure destination folder exists before saving
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Scale signal below clipping threshold with 10% safety margin
    normalized = signal / (np.max(np.abs(signal)) * 1.1)
    # Convert normalized float values to 16-bit signed PCM
    pcm16 = (normalized * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, pcm16)


def plot_signals(S, X, S_aligned, path):
    """Plot a 3x3 visual comparison of true sources, mixtures, and ICA recovered outputs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
    column_data = [S, X, S_aligned]
    titles = ["Original sources", "Mixed signals", "Recovered by ICA"]

    for col_idx, (col_signals, col_title) in enumerate(zip(column_data, titles)):
        axes[0, col_idx].set_title(col_title)
        for row_idx in range(3):
            # Display time-domain waveform for each channel row
            axes[row_idx, col_idx].plot(col_signals[row_idx], color="tab:blue", lw=0.6)
            axes[row_idx, col_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_correlation(corr_matrix, path):
    """Render a labeled heatmap of absolute cross-correlations between sources and outputs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(corr_matrix, cmap="Blues", vmin=0, vmax=1)
    plt.title("Source vs ICA output |correlation|\n(bright diagonal after reordering = good)", fontsize=9)
    plt.colorbar()

    # Overlay numeric correlation values in each grid cell
    for row in range(corr_matrix.shape[0]):
        for col in range(corr_matrix.shape[1]):
            val = corr_matrix[row, col]
            color = "white" if val > 0.5 else "black"
            plt.text(col, row, f"{val:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    """Execute the full 3-source mixture generation and FastICA separation pipeline."""
    # a. Load source tracks and simulate linear audio mixture
    S, sample_rate = load_sources(SOURCE_FILES)
    # X shape: (3, n_samples)
    X = MIXING_MATRIX @ S
    for idx in range(3):
        save_wav(f"mixtures/mix_{idx + 1}.wav", X[idx], sample_rate)

    # b. Run FastICA pipeline: centering, whitening, and iterative unmixing
    X_centered, _ = center(X)
    Z, _ = whiten(X_centered)
    S_est, _ = fastica(Z)

    # d. Match estimated sources with reference signals to fix permutation & sign ambiguities
    results = match_and_score(S, S_est)
    S_aligned = results["S_aligned"]

    # c. Display kurtosis comparison demonstrating the Central Limit Theorem
    print(f"\n{'Signal':<10} | {'Source Kurtosis':<16} | {'Mixture Kurtosis':<17} | {'Recovered Kurtosis':<18}")
    print("-" * 68)
    for idx in range(3):
        k_src = kurtosis(S[idx])
        k_mix = kurtosis(X[idx])
        k_rec = kurtosis(S_aligned[idx])
        print(f"Track {idx + 1:<4} | {k_src:<16.3f} | {k_mix:<17.3f} | {k_rec:<18.3f}")

    # Print separation metrics
    print("\n--- Separation Performance ---")
    for idx in range(3):
        print(f"Source {idx + 1}: |Corr| = {results['abs_correlations'][idx]:.4f}, SDR = {results['sdr_db'][idx]:.2f} dB")
    print(f"Mean SDR: {np.mean(results['sdr_db']):.2f} dB\n")

    # e. Export aligned separated waveforms
    for idx, file_path in enumerate(SOURCE_FILES):
        name = os.path.splitext(os.path.basename(file_path))[0]
        save_wav(f"separated/recovered_{name}.wav", S_aligned[idx], sample_rate)

    # f. Generate waveform and correlation visual figures
    plot_signals(S, X, S_aligned, "results/signals.png")
    plot_correlation(results["corr_matrix"], "results/correlation.png")


if __name__ == "__main__":
    main()
