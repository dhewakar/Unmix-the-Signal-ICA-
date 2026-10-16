# Unmix the Signal: ICA from scratch

## 1. Overview
This project demonstrates how to unmix overlapping audio signals and isolate individual sound sources using Independent Component Analysis (ICA) implemented strictly from mathematical first principles. By leveraging non-Gaussian statistical distributions and fixed-point optimization in NumPy, the algorithm solves the blind source separation "cocktail party problem" without external machine learning libraries. An interactive Streamlit dashboard accompanies the mathematical engine, offering audio playback, diagnostic visualizations, and a landmark audio fingerprinting recognizer.

![App screenshot](examples/screenshot.png)

## 2. Project Structure
- `ica.py`: Core math library implementing centering, manual PCA whitening, symmetric FastICA with history tracking, kurtosis, and Hungarian matching metrics.
- `shazam_mini.py`: Audio fingerprinting module creating spectral peak hashes and performing time-offset histogram voting for track recognition.
- `app.py`: Full-featured Streamlit interactive web application with custom dark theme, audio format conversion, and real-time visualization tabs.
- `requirements.txt`: Minimal dependencies (`numpy`, `scipy`, `matplotlib`, `streamlit`, `imageio-ffmpeg`).

## 3. How to Run
Website Link: https://unmixthesignal.streamlit.app/

## 4. How to Use the Website
- **Mode A (Clean Sources):** Upload 2 to 4 separate tracks (e.g., speech, music, ambient). The app generates a well-conditioned mixing matrix $A$, blends them, runs FastICA, and evaluates recovery against ground truth.
- **Mode B (Real Mixed Recordings):** Upload 2 to 4 real-world microphone mixtures. The app decomposes them directly into independent components without needing ground truth.
- **Tabs:**
  1. *Separate:* Audio players for sources, mixtures, and separated tracks, plus SDR metrics and waveform grids.
  2. *PCA vs ICA:* Direct comparison showing why second-order decorrelation fails to isolate sources while FastICA succeeds.
  3. *Under the Hood:* Covariance matrices before/after whitening, 2D scatter projections, kurtosis bars, and optimization curves.
  4. *Noise Sweep:* Interactive benchmark testing separation quality against additive Gaussian noise from 40 dB to 0 dB.
  5. *Shazam:* Audio fingerprint library builder and recognizer capable of identifying noisy slices or separated audio tracks.
  6. *Concepts:* Educational deep-dive explainers with intuitive physical analogies.

## 5. Audio Sources
| File | Type | Link | License |
| :--- | :--- | :--- | :--- |
| `speech.wav` | Speech | [PASTE LINK] | [PASTE LICENSE] |
| `music.wav` | Music | [PASTE LINK] | [PASTE LICENSE] |
| `ambient.wav` | Ambient | [PASTE LINK] | [PASTE LICENSE] |

## 6. Audio Files
- **Original Sources:** [`examples/sources/`](examples/sources/)
- **Linear Mixtures:** [`examples/mixtures/`](examples/mixtures/)
- **Separated Outputs:** [`examples/separated/`](examples/separated/)
- **10 dB Noisy Samples:** [`examples/noisy10dB/`](examples/noisy10dB/)

## 7. Pipeline
1. **Mix Sources:** $X = A S$, where $S$ has shape $(n, T)$ and linear mixtures $X$ have shape $(n, T)$.
2. **Center Signals:** $X_{\text{centered}} = X - \mu$, shape $(n, T)$, zeroing mean expectation across time.
3. **PCA Whitening:** $Z = \Lambda^{-1/2} E^T X_{\text{centered}}$, shape $(n, T)$, transforming data to have identity covariance $I$.
4. **FastICA Optimization:** Iteratively solve for orthogonal unmixing matrix $W$ $(n, n)$ to compute independent sources $S_{\text{est}} = W Z$, shape $(n, T)$.
5. **Ambiguity Resolution:** Normalize variance, resolve sign flips, and pair components to ground truth via Hungarian matching.
6. **Export & Evaluation:** Stream audio players, export WAV/ZIP archives, and compute Signal-to-Distortion Ratio (SDR) and correlations.

## 8. Concepts
### a. Why ICA over PCA
PCA only analyzes second-order statistics (covariance), identifying perpendicular axes that maximize variance. Two signals can be completely uncorrelated while remaining statistically dependent. Because whitened data has a spherical covariance matrix, every possible orthogonal rotation produces equally "uncorrelated" outputs. ICA looks beyond covariance to higher-order statistics (kurtosis and negentropy) to find the single correct rotation that achieves genuine statistical independence.
*Analogy:* PCA aligns a circular pizza to its outer crust, but cannot tell which slice is which; ICA looks at the toppings to orient each slice correctly.

### b. What Whitening Does and Why It Simplifies ICA
Whitening normalizes variances to 1 and eliminates all linear correlations, turning the covariance matrix into the identity matrix $I$. In geometrical terms, whitening removes stretching and shearing from the mixing matrix $A$, transforming the original coordinate cloud into a symmetric sphere. This simplifies the ICA problem because the remaining unknown mixing transformation is purely a rigid rotation (an orthogonal matrix). Consequently, the search space collapses from $n \times n$ free variables down to $\frac{n(n-1)}{2}$ rotational angles.
*Analogy:* Whitening reshapes a stretched ellipse into a perfect basketball, so all that is left to do is rotate the ball until the logos align.

### c. Why Non-Gaussianity Enables Separation (Central Limit Theorem)
The Central Limit Theorem dictates that the sum of independent non-Gaussian variables is always more Gaussian than any individual variable. Therefore, linear audio mixtures are closer to a normal bell-curve distribution than the original distinct sound sources. FastICA exploits this principle in reverse: by rotating the unmixing vectors to maximize non-Gaussianity (measured using excess kurtosis and tanh contrast), the algorithm isolates the individual components. A Gaussian source has zero directional preference in higher moments, so rotating a Gaussian cloud changes nothing; thus, at most one Gaussian source can be separated.
*Analogy:* Blending colorful fruits creates a uniform brown smoothie; separating them requires searching for the pure, unblended red, yellow, and green colors.

### d. How Experiments and Metrics Were Chosen
Because blind source separation cannot recover original amplitude or channel order ($X = A S = (A P^{-1})(P S)$), metrics must handle scale, sign, and permutation ambiguities. The evaluation standardizes signals to unit variance, uses the Hungarian algorithm (`linear_sum_assignment`) on absolute cross-correlations to pair tracks, and flips negative correlations. It then computes the Signal-to-Distortion Ratio (SDR) in decibels using the optimal scalar projection $\alpha = \frac{\langle s_{\text{est}}, s_{\text{true}}\rangle}{\|s_{\text{true}}\|^2}$. The PCA-vs-ICA experiment proves that decorrelation alone cannot unmix signals, while the SNR sweep tests resilience against ambient recording noise.

## 9. Training / Metric Graphs
Because ICA is an unsupervised learning method, there are no ground-truth training labels or validation loss functions during optimization:
1. **Convergence Error:** Tracks $\| | \text{diag}(W_{\text{new}} W^T) | - 1 \|_{\infty}$, displaying step-by-step decay towards zero on a log scale.
2. **Objective $J(y)$:** Evaluates negentropy approximated via $(\mathbb{E}[\log \cosh(w^T z)] - 0.3746)^2$; increases as components become less Gaussian.
3. **Loss ($-J$):** The negated objective function that decreases monotonically as the fixed-point optimization converges.
4. **Mean Correlation Curve:** Ground-truth validation curve (Mode A only) showing how true source reconstruction accuracy improves per iteration.

## 10. Results
- **Run ICA Mean SDR:** `[PASTE: mean SDR]` dB

### PCA vs. ICA Comparison
| Source | PCA \|corr\| | PCA SDR (dB) | ICA \|corr\| | ICA SDR (dB) |
| :--- | :--- | :--- | :--- | :--- |
| Track 1 | `[PASTE]` | `[PASTE]` | `[PASTE]` | `[PASTE]` |
| Track 2 | `[PASTE]` | `[PASTE]` | `[PASTE]` | `[PASTE]` |
| Track 3 | `[PASTE]` | `[PASTE]` | `[PASTE]` | `[PASTE]` |

### SNR Robustness Sweep
| Input SNR (dB) | Output Mean SDR (dB) | Output Mean \|corr\| |
| :--- | :--- | :--- |
| 40 | `[PASTE]` | `[PASTE]` |
| 30 | `[PASTE]` | `[PASTE]` |
| 20 | `[PASTE]` | `[PASTE]` |
| 10 | `[PASTE]` | `[PASTE]` |
| 5 | `[PASTE]` | `[PASTE]` |
| 0 | `[PASTE]` | `[PASTE]` |

### Visual Figures
- Waveform Comparisons: `examples/signals.png`
- FastICA Training Progress: `examples/training.png`
- PCA vs. ICA Benchmarks: `examples/pca_vs_ica.png`
- Noise Robustness Curve: `examples/snr_sweep.png`

## 11. Mini-Shazam
1. Computes STFT spectrograms with a Hann window and picks the highest spectral peak in 4 frequency bands per frame.
2. Forms landmark hashes $(f_1, f_2, \Delta t)$ by pairing each peak with the next 3 peaks within 64 frames.
3. Stores hashes in an inverted dictionary mapping each unique pattern to `(song_name, anchor_time)`.
4. Queries search the dictionary and bin matched timestamps into a histogram of time offsets: $\Delta = t_{\text{song}} - t_{\text{clip}}$.
5. The song with the highest coherent vote count at a single consistent time offset wins the match.

## 12. Limitations
1. **Instantaneous Mixing Only:** Assumes signals arrive at microphones instantaneously without room reverberation, echoes, or acoustic propagation delays.
2. **Determined / Overdetermined Requirement:** Requires at least as many recording mixtures as source signals ($m \ge n$); underdetermined setups cannot be unmixed with standard matrix inversion.
3. **Random Initialization Sensitivity:** FastICA utilizes random orthogonal starts, meaning independent components can converge in different channel permutations across runs.
4. **Noise Vulnerability:** High additive Gaussian noise reduces higher-order contrast, leading to lower SDR separation at low SNR levels.
