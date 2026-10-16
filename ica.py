"""Independent Component Analysis (ICA) implemented from scratch using NumPy."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def center(X):
    """Subtract the mean of each signal row to center the data at zero."""
    # X shape: (n_signals, n_samples)
    row_means = np.mean(X, axis=1, keepdims=True)
    X_centered = X - row_means
    return X_centered, row_means


def whiten(X_centered):
    """Linearly transform centered data so components are uncorrelated with unit variance."""
    # X_centered shape: (n_signals, n_samples)
    n_samples = X_centered.shape[1]
    cov = (X_centered @ X_centered.T) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    whitening_matrix = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    # Z shape: (n_signals, n_samples)
    Z = whitening_matrix @ X_centered
    # After this, np.cov(Z) is approximately the identity matrix
    return Z, whitening_matrix


def symmetric_decorrelation(W):
    """Orthogonalize unmixing matrix W symmetrically to prevent component collapse."""
    # W shape: (n_components, n_components)
    eigenvalues, eigenvectors = np.linalg.eigh(W @ W.T)
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    # This keeps the rows of W orthogonal so components don't collapse onto the same source
    return inv_sqrt @ W


def fastica(Z, max_iter=500, tol=1e-6):
    """Estimate independent sources and track convergence and negentropy history."""
    # Z shape: (n_components, n_samples)
    np.random.seed(0)
    n_components, n_samples = Z.shape

    W = np.random.randn(n_components, n_components)
    W = symmetric_decorrelation(W)

    history = {"conv": [], "objective": [], "W": []}

    for iteration in range(max_iter):
        WZ = W @ Z
        g_WZ = np.tanh(WZ)
        g_prime_WZ = 1.0 - g_WZ**2

        term1 = (g_WZ @ Z.T) / n_samples
        term2 = np.diag(np.mean(g_prime_WZ, axis=1)) @ W
        W_new = symmetric_decorrelation(term1 - term2)

        convergence_metric = float(np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)))

        # This approximates negentropy J(y). 0.3746 is E[log cosh(g)] for a standard Gaussian g.
        # Bigger = more non-Gaussian = what FastICA is maximizing.
        wz_new = W_new @ Z
        negentropy_obj = float(np.mean((np.mean(np.log(np.cosh(wz_new)), axis=1) - 0.3746) ** 2))

        history["conv"].append(convergence_metric)
        history["objective"].append(negentropy_obj)
        history["W"].append(np.copy(W_new))

        W = W_new
        if convergence_metric < tol:
            break

    S_estimated = W @ Z
    return S_estimated, W, history


def kurtosis(x):
    """Compute excess kurtosis of a 1D array to measure non-Gaussianity."""
    mean_val = np.mean(x)
    std_val = np.std(x)
    # Gaussian = 0; this is the measure of non-Gaussianity used to explain why ICA works
    return np.mean(((x - mean_val) / std_val) ** 4) - 3.0


def match_and_score(S_true, S_estimated):
    """Resolve permutation, sign, and scale ambiguities and calculate SDR scores."""
    n_true, n_samples = S_true.shape
    n_est = S_estimated.shape[0]

    # Step 1: Normalize every row to zero mean and unit variance (fixes scale)
    S_true_norm = (S_true - np.mean(S_true, axis=1, keepdims=True)) / np.std(S_true, axis=1, keepdims=True)
    S_est_norm = (S_estimated - np.mean(S_estimated, axis=1, keepdims=True)) / np.std(S_estimated, axis=1, keepdims=True)

    # Step 2: Sample cross-correlation matrix, shape: (n_true, n_est)
    corr = (S_true_norm @ S_est_norm.T) / n_samples

    # Step 3: Optimal bipartite matching to pair each true source with an estimate (fixes permutation)
    true_indices, est_indices = linear_sum_assignment(-np.abs(corr))

    # Step 4 & 5: Align signals, correct sign, and compute SDR in decibels
    S_aligned = np.zeros((n_true, n_samples))
    abs_correlations = []
    sdr_db = []

    for true_idx, est_idx in zip(true_indices, est_indices):
        aligned_signal = S_estimated[est_idx].copy()
        current_corr = corr[true_idx, est_idx]

        if current_corr < 0:
            aligned_signal = -aligned_signal

        S_aligned[true_idx] = aligned_signal
        abs_correlations.append(float(np.abs(current_corr)))

        ref_signal = S_true[true_idx]
        alpha = np.dot(aligned_signal, ref_signal) / np.dot(ref_signal, ref_signal)
        target_component = alpha * ref_signal
        distortion = aligned_signal - target_component
        sdr_val = 10.0 * np.log10(np.sum(target_component**2) / np.sum(distortion**2))
        sdr_db.append(float(sdr_val))

    return {
        "S_aligned": S_aligned,
        "abs_correlations": abs_correlations,
        "sdr_db": sdr_db,
        "corr_matrix": np.abs(corr),
    }


def make_mixing_matrix(n, seed=0):
    """Generate well-conditioned random square mixing matrix with condition number < 30."""
    rng = np.random.RandomState(seed)
    while True:
        # A badly conditioned A makes separation very hard, so reject until cond(A) < 30
        A = rng.uniform(0.3, 1.0, (n, n)) + np.eye(n)
        if np.linalg.cond(A) < 30:
            return A


def add_noise(X, snr_db, seed=0):
    """Add row-wise Gaussian noise scaled to the target SNR level in decibels."""
    # X shape: (n_signals, n_samples)
    rng = np.random.RandomState(seed)
    signal_power = np.mean(X**2, axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0, 1, X.shape) * np.sqrt(noise_power)
    return X + noise


def run_ica(X):
    """Execute complete FastICA pipeline returning components, unmixing matrices, and covariances."""
    X_centered, _ = center(X)
    Z, _ = whiten(X_centered)
    S_est, W, history = fastica(Z)
    return {
        "X_centered": X_centered,
        "Z": Z,
        "S_est": S_est,
        "W": W,
        "history": history,
        "cov_before": np.cov(X_centered),
        "cov_after": np.cov(Z),
    }


def pca_vs_ica(S_true, ica_out):
    """Compare PCA whitening baseline directly against FastICA source recovery."""
    # PCA output = whitened data Z = uncorrelated but not independent
    return {
        "pca": match_and_score(S_true, ica_out["Z"]),
        "ica": match_and_score(S_true, ica_out["S_est"]),
    }


def corr_history(S_true, Z, W_list):
    """Calculate mean absolute correlation at each unmixing step during training."""
    # This is the "accuracy per iteration" training curve
    history_corrs = []
    for W in W_list:
        S_est = W @ Z
        score = match_and_score(S_true, S_est)
        history_corrs.append(float(np.mean(score["abs_correlations"])))
    return history_corrs


def snr_sweep(S, A, snr_list=(40, 30, 20, 10, 5, 0)):
    """Evaluate FastICA separation performance across a series of SNR degradation levels."""
    X_clean = A @ S
    sweep_results = []
    for snr in snr_list:
        X_noisy = add_noise(X_clean, snr)
        out = run_ica(X_noisy)
        res = match_and_score(S, out["S_est"])
        sweep_results.append((snr, res))
    return sweep_results


if __name__ == "__main__":
    np.random.seed(0)
    num_samples = 8000
    time = np.linspace(0.0, 1.0, num_samples, endpoint=False)

    s1 = np.sin(2.0 * np.pi * 5.0 * time)
    s2 = 2.0 * (time * 7.0 - np.floor(time * 7.0)) - 1.0
    S_true = np.vstack([s1, s2])

    A = make_mixing_matrix(2, seed=0)
    X = A @ S_true

    out = run_ica(X)
    results = match_and_score(S_true, out["S_est"])

    print("Iterations:", len(out["history"]["conv"]))
    print("Absolute Correlations:", results["abs_correlations"])
    print("SDR (dB):", results["sdr_db"])
