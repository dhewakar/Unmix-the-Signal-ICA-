"""Streamlit dashboard for Independent Component Analysis (ICA) from scratch."""

import io
import os
import subprocess
import tempfile
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from ica import (
    add_noise,
    corr_history,
    kurtosis,
    make_mixing_matrix,
    match_and_score,
    pca_vs_ica,
    run_ica,
    snr_sweep,
)
from shazam_mini import build_database, identify

SAMPLE_RATE = 22050
AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "mp4", "webm"]
COLORS = ["#E2FF66", "#A259FF", "#2DD4A7", "#FF8A3D"]


# STYLING HELPERS
def inject_css():
    """Inject bespoke dark theme stylesheet with typography and custom components."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
      --bg: #12171F;
      --card: #1B2330;
      --line: #2A3544;
      --text: #E8EAED;
      --muted: #9AA4B2;
      --lime: #E2FF66;
      --purple: #A259FF;
      --teal: #2DD4A7;
      --grad: linear-gradient(120deg, #B500B2 0%, #F5007A 40%, #F57C00 75%, #F5D800 100%);
    }

    .stApp, body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Space Grotesk', sans-serif;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {
      display: none !important;
    }

    .block-container {
      max-width: 1200px;
      padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] {
      background: #0D1117;
      border-right: 1px solid var(--line);
    }

    .stButton > button {
      background: var(--lime) !important;
      color: #12171F !important;
      font-weight: 700 !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 0.7rem 1.4rem !important;
      position: relative !important;
      overflow: hidden !important;
      transition: transform 0.15s ease, filter 0.15s ease !important;
    }
    .stButton > button:hover {
      filter: brightness(1.08) !important;
      transform: translateY(-1px) !important;
    }
    .stButton > button::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 26px;
      height: 26px;
      background: linear-gradient(135deg, rgba(0,0,0,0.25) 50%, transparent 50%);
      pointer-events: none;
    }

    .stDownloadButton > button {
      background: var(--purple) !important;
      color: #FFFFFF !important;
      font-weight: 700 !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 0.7rem 1.4rem !important;
      position: relative !important;
      overflow: hidden !important;
      transition: transform 0.15s ease, filter 0.15s ease !important;
    }
    .stDownloadButton > button:hover {
      filter: brightness(1.08) !important;
      transform: translateY(-1px) !important;
    }
    .stDownloadButton > button::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 26px;
      height: 26px;
      background: linear-gradient(135deg, rgba(0,0,0,0.25) 50%, transparent 50%);
      pointer-events: none;
    }

    [data-baseweb="tab-list"] {
      gap: 8px !important;
    }
    [data-baseweb="tab"] {
      background: var(--card) !important;
      border-radius: 999px !important;
      padding: 8px 18px !important;
      color: var(--muted) !important;
      border: none !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
      background: var(--lime) !important;
      color: #12171F !important;
      font-weight: 700 !important;
    }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
      display: none !important;
    }

    [data-testid="stFileUploader"] {
      background: var(--grad);
      border-radius: 28px;
      padding: 28px;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p {
      color: #FFFFFF !important;
      font-weight: 700;
      font-size: 1.1rem;
    }
    [data-testid="stFileUploaderDropzone"] {
      background: #FFFFFF !important;
      border-radius: 22px !important;
      border: 2px dashed #cbd5e1 !important;
      min-height: 170px !important;
      color: #1F2937 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
      color: #1F2937 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
      background: #12171F !important;
      color: #FFFFFF !important;
      border-radius: 999px !important;
      border: none !important;
      padding: 8px 18px !important;
    }

    [data-testid="stMetric"] {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: var(--lime) !important;
      font-size: 2rem !important;
    }

    [data-testid="stExpander"] {
      background: var(--card) !important;
      border-radius: 14px !important;
      border: 1px solid var(--line) !important;
    }

    audio {
      width: 100%;
    }
    [data-testid="stDataFrame"] {
      border-radius: 12px;
      overflow: hidden;
    }
    [data-testid="stAlert"] {
      border-radius: 14px !important;
    }

    .hero {
      position: relative;
      overflow: hidden;
      border-radius: 28px;
      padding: 56px 48px;
      min-height: 380px;
      background: linear-gradient(180deg, #161C26, #12171F);
      border: 1px solid var(--line);
      margin-bottom: 24px;
    }
    .hero > * {
      position: relative;
      z-index: 1;
    }
    .hero h1 {
      font-size: clamp(2.8rem, 7vw, 5.5rem);
      line-height: 1.02;
      letter-spacing: -0.02em;
      margin: 24px 0 16px;
    }
    .lime {
      background: linear-gradient(90deg, #E2FF66, #F1F5C8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: transparent;
    }
    .grey {
      color: #C9CDD3;
    }
    .hero p {
      color: var(--muted);
      font-size: 1.2rem;
      max-width: 560px;
    }
    .pill {
      font-family: 'JetBrains Mono', monospace;
      color: var(--lime);
      border: 1px solid var(--lime);
      border-radius: 999px;
      padding: 4px 14px;
      margin-right: 8px;
      display: inline-block;
      font-size: 0.85rem;
    }
    .pill-outline {
      font-family: 'JetBrains Mono', monospace;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 14px;
      margin-right: 8px;
      display: inline-block;
      font-size: 0.85rem;
    }
    .blob {
      position: absolute;
      border-radius: 50%;
      filter: blur(70px);
      opacity: 0.55;
      z-index: 0;
    }
    .blob-purple {
      width: 320px;
      height: 320px;
      background: #7C3AED;
      right: 8%;
      top: 8%;
    }
    .blob-teal {
      width: 220px;
      height: 220px;
      background: #10B981;
      right: 2%;
      bottom: 6%;
    }
    .beam {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 55%;
      height: 45%;
      z-index: 0;
      background: linear-gradient(to top right, rgba(226, 255, 102, 0.55), transparent 70%);
      clip-path: polygon(0% 100%, 60% 20%, 100% 20%, 100% 100%);
    }

    .steps {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 28px;
    }
    .step-card {
      flex: 1 1 170px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      transition: transform 0.2s ease;
    }
    .step-card:hover {
      transform: translateY(-3px);
    }
    .step-num {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      background: var(--grad);
      color: #FFFFFF;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 10px;
    }
    .step-card small {
      color: var(--muted);
    }

    h2.sec {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
      color: var(--lime);
      margin: 28px 0 8px;
    }

    @media (max-width: 800px) {
      .hero {
        padding: 32px 22px;
      }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def hero():
    """Render the hero showcase card with glow effects and badge tags."""
    html = """
    <div class="hero">
      <div class="blob blob-purple"></div>
      <div class="blob blob-teal"></div>
      <div class="beam"></div>
      <div>
        <span class="pill">&lt;upload&gt;</span>
        <span class="pill">&lt;mix&gt;</span>
        <span class="pill">&lt;whiten&gt;</span>
        <span class="pill">&lt;fastica&gt;</span>
        <span class="pill">&lt;evaluate&gt;</span>
      </div>
      <h1><span class="lime">Unmix</span> <span class="grey">the Signal</span></h1>
      <p>Drop in mixed audio and get the clean sources back. Independent Component Analysis, written from scratch with NumPy.</p>
      <div>
        <span class="pill-outline">any audio format</span>
        <span class="pill-outline">PCA whitening</span>
        <span class="pill-outline">FastICA</span>
        <span class="pill-outline">mini-Shazam</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def steps_row():
    """Render 5-stage conceptual workflow cards."""
    html = """
    <div class="steps">
      <div class="step-card">
        <div class="step-num">1</div>
        <b>Upload</b><br>
        <small>any format to WAV</small>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <b>Mix</b><br>
        <small>X = A·S</small>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <b>Whiten</b><br>
        <small>covariance = identity</small>
      </div>
      <div class="step-card">
        <div class="step-num">4</div>
        <b>FastICA</b><br>
        <small>learn W</small>
      </div>
      <div class="step-card">
        <div class="step-num">5</div>
        <b>Evaluate</b><br>
        <small>correlation and SDR</small>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_title(text):
    """Render monospace section headers in lime accent."""
    st.markdown(f'<h2 class="sec">&lt;{text}&gt;</h2>', unsafe_allow_html=True)


# A) AUDIO HELPERS
@st.cache_data
def convert_audio(file_bytes, filename, max_seconds):
    """Convert any uploaded audio into mono float32 WAV at SAMPLE_RATE using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ext = os.path.splitext(filename)[1] or ".wav"
        in_path, out_path = os.path.join(tmp_dir, f"in{ext}"), os.path.join(tmp_dir, "out.wav")
        with open(in_path, "wb") as f:
            f.write(file_bytes)
        exe = imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else "ffmpeg"
        cmd = [exe, "-y", "-i", in_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-t", str(max_seconds), out_path]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            raise ValueError(f"Could not read {filename}. Is it a valid audio file?")
        _, data = wavfile.read(out_path)
        return (data / 32768.0).astype(np.float32) if np.issubdtype(data.dtype, np.integer) else data.astype(np.float32)


def load_signals(uploaded_files, max_seconds):
    """Load, convert, and trim multiple audio signals to equal sample length."""
    sigs = [convert_audio(f.getvalue(), f.name, max_seconds) for f in uploaded_files]
    names = [os.path.splitext(f.name)[0] for f in uploaded_files]
    min_len = min(len(s) for s in sigs)
    # signals shape: (n_signals, n_samples)
    return np.array([s[:min_len] for s in sigs], dtype=np.float32), names


def to_wav_bytes(signal):
    """Normalize and encode 1D audio array to 16-bit PCM WAV bytes."""
    peak = max(float(np.max(np.abs(signal))), 1e-9)
    pcm = (signal / (peak * 1.1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, pcm)
    return buf.getvalue()


def audio_row(signals, names, key_prefix):
    """Display horizontal grid of players and download buttons for audio tracks."""
    cols = st.columns(len(signals))
    for i, (sig, name) in enumerate(zip(signals, names)):
        with cols[i]:
            st.caption(name)
            wav_bytes = to_wav_bytes(sig)
            st.audio(wav_bytes, format="audio/wav")
            st.download_button("Download WAV", wav_bytes, f"{key_prefix}_{name}.wav", "audio/wav", key=f"{key_prefix}_{i}")


def make_zip(files_dict):
    """Bundle dictionary of filenames and signals into an in-memory ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, sig in files_dict.items():
            zf.writestr(fname, to_wav_bytes(sig))
    return buf.getvalue()


# B) PLOT HELPERS
def apply_plot_theme():
    """Apply consistent dark styling across all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor": "#161B22", "axes.facecolor": "#161B22",
        "text.color": "#E8EAED", "axes.labelcolor": "#E8EAED",
        "xtick.color": "#E8EAED", "ytick.color": "#E8EAED",
        "grid.color": "#2A3544", "axes.grid": True,
    })


def plot_signal_grid(rows):
    """Plot multi-stage waveforms across signals, downsampled for rendering speed."""
    n_rows, n_cols = len(rows), len(rows[0][1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 1.8 * n_rows), sharex=True, squeeze=False)
    for r, (title, sigs) in enumerate(rows):
        for c in range(n_cols):
            step = max(1, len(sigs[c]) // 2000)
            axes[r, c].plot(sigs[c][::step], color=COLORS[c % len(COLORS)], lw=0.7)
            if c == 0:
                axes[r, c].set_ylabel(title, fontsize=8)
    fig.tight_layout()
    return fig


def plot_heatmap(corr_matrix, names):
    """Plot cross-correlation heatmap with overlay numeric text values."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(corr_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("|correlation| before re-ordering", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(corr_matrix.shape[1]))
    ax.set_xticklabels([f"ICA {i+1}" for i in range(corr_matrix.shape[1])], fontsize=8)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_scatter(panels):
    """Scatter 2D projections of component 0 vs 1 across separation stages."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3), squeeze=False)
    for idx, (title, data) in enumerate(panels):
        ax = axes[0, idx]
        # data shape: (n_signals, n_samples)
        pts = min(2000, data.shape[1])
        sub = data[:, np.random.RandomState(0).choice(data.shape[1], pts, replace=False)]
        ax.scatter(sub[0], sub[1], s=2, alpha=0.4, color=COLORS[idx % len(COLORS)])
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_kurtosis(groups):
    """Bar chart comparing excess kurtosis between signal processing stages."""
    fig, ax = plt.subplots(figsize=(6, 3))
    group_names = list(groups.keys())
    n_sigs = len(groups[group_names[0]])
    x = np.arange(n_sigs)
    w = 0.8 / len(groups)
    for i, (g_name, sigs) in enumerate(groups.items()):
        kurts = [kurtosis(sigs[k]) for k in range(n_sigs)]
        ax.bar(x + i * w - 0.4 + w / 2, kurts, w, label=g_name, color=COLORS[i % len(COLORS)])
    ax.set_title("Excess kurtosis (0 = Gaussian)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Track {i+1}" for i in range(n_sigs)])
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_training(history, corr_curve):
    """Display 2x2 training history: convergence, negentropy objective, loss, and correlation."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))
    iters = np.arange(1, len(history["conv"]) + 1)
    axes[0, 0].semilogy(iters, history["conv"], color=COLORS[0], marker=".")
    axes[0, 0].set_title("Convergence Error (log scale)", fontsize=8)
    axes[0, 1].plot(iters, history["objective"], color=COLORS[1], marker=".")
    axes[0, 1].set_title("Objective J(y) (Negentropy)", fontsize=8)
    axes[1, 0].plot(iters, -np.array(history["objective"]), color=COLORS[2], marker=".")
    axes[1, 0].set_title("Loss (-Objective)", fontsize=8)
    if corr_curve is not None:
        axes[1, 1].plot(iters, corr_curve, color=COLORS[3], marker=".")
        axes[1, 1].set_title("Mean |corr| vs Iteration", fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "needs clean sources (Mode A)", ha="center", va="center", fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Iteration", fontsize=7)
    fig.tight_layout()
    return fig


def plot_bars(names, pca_res, ica_res):
    """Grouped bar plots comparing SDR and correlation for PCA vs FastICA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    x, w = np.arange(len(names)), 0.35
    ax1.bar(x - w / 2, pca_res["sdr_db"], w, label="PCA", color=COLORS[1])
    ax1.bar(x + w / 2, ica_res["sdr_db"], w, label="ICA", color=COLORS[0])
    ax1.set_title("SDR (dB)", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8); ax1.legend(fontsize=8)
    ax2.bar(x - w / 2, pca_res["abs_correlations"], w, label="PCA", color=COLORS[1])
    ax2.bar(x + w / 2, ica_res["abs_correlations"], w, label="ICA", color=COLORS[0])
    ax2.set_title("|correlation|", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sweep(sweep):
    """Plot separation accuracy and SDR degradation curves across input SNR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    snrs = [s[0] for s in sweep]
    sdrs = [float(np.mean(s[1]["sdr_db"])) for s in sweep]
    corrs = [float(np.mean(s[1]["abs_correlations"])) for s in sweep]
    ax1.plot(snrs, sdrs, marker="o", color=COLORS[0], lw=1.5)
    ax1.set_title("Mean SDR vs Input SNR", fontsize=9); ax1.set_xlabel("Input SNR (dB)")
    ax2.plot(snrs, corrs, marker="s", color=COLORS[2], lw=1.5)
    ax2.set_title("Mean |corr| vs Input SNR", fontsize=9); ax2.set_xlabel("Input SNR (dB)")
    fig.tight_layout()
    return fig


def plot_scores(scores):
    """Horizontal bar chart showing Shazam alignment vote scores per candidate track."""
    fig, ax = plt.subplots(figsize=(6, max(2.0, len(scores) * 0.45)))
    songs, votes = list(scores.keys()), list(scores.values())
    max_v = max(votes) if votes else 0
    bar_colors = [COLORS[0] if (v == max_v and max_v > 0) else COLORS[1] for v in votes]
    ax.barh(songs, votes, color=bar_colors)
    ax.set_title("Shazam Candidate Coherent Offset Votes", fontsize=9)
    ax.set_xlabel("Votes at Single Best Offset", fontsize=8)
    fig.tight_layout()
    return fig


# C) PIPELINE
def run_separation(files, mode, max_seconds, seed, snr_db):
    """Run FastICA unmixing pipeline in Mode A (synthetic mixing) or Mode B (real recordings)."""
    st.session_state.pop("sweep", None)
    if "A:" in mode:
        # S shape: (n_signals, n_samples)
        S, names = load_signals(files, max_seconds)
        A = make_mixing_matrix(len(S), seed)
        X_clean = A @ S
        X = add_noise(X_clean, snr_db, seed) if snr_db < 60 else X_clean
        out = run_ica(X)
        score = match_and_score(S, out["S_est"])
        pca = pca_vs_ica(S, out)
        curve = corr_history(S, out["Z"], out["history"]["W"])
        st.session_state["res"] = {
            "mode": "A", "names": names, "S": S, "A": A, "X_clean": X_clean,
            "X": X, "out": out, "score": score, "pca": pca, "curve": curve,
        }
    else:
        X, names = load_signals(files, max_seconds)
        out = run_ica(X)
        st.session_state["res"] = {
            "mode": "B", "names": names, "S": None, "A": None, "X_clean": None,
            "X": X, "out": out, "score": None, "pca": None, "curve": None,
        }


# E) TABS
def tab_separate(res):
    """Render separated source audio, metrics, waveforms, and ZIP downloads."""
    out, names, X = res["out"], res["names"], res["X"]
    if res["mode"] == "A":
        S, A, score = res["S"], res["A"], res["score"]
        section_title("1. Sources converted to WAV"); audio_row(S, names, "src")
        section_title("2. Mixtures X = A @ S"); st.dataframe(A.round(2)); audio_row(X, [f"Mix {i+1}" for i in range(len(X))], "mix")
        section_title("3. Recovered by ICA"); audio_row(score["S_aligned"], names, "rec")
        st.caption("ICA can return sources in any order, sign and volume; we re-order, flip sign and rescale before scoring.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean |correlation|", f"{np.mean(score['abs_correlations']):.4f}")
        c2.metric("Mean SDR (dB)", f"{np.mean(score['sdr_db']):.2f}")
        c3.metric("FastICA iterations", len(out["history"]["conv"]))
        st.dataframe([{"Source": n, "|correlation|": round(c, 4), "SDR (dB)": round(s, 2)}
                      for n, c, s in zip(names, score["abs_correlations"], score["sdr_db"])])
        fig1 = plot_signal_grid([("Original", S), ("Mixed", X), ("Recovered", score["S_aligned"])])
        st.pyplot(fig1); plt.close(fig1)
        fig2 = plot_heatmap(score["corr_matrix"], names)
        st.pyplot(fig2); plt.close(fig2)
        files_map = {f"mix_{i+1}.wav": X[i] for i in range(len(X))}
        files_map.update({f"recovered_{n}.wav": score["S_aligned"][i] for i, n in enumerate(names)})
        st.download_button("Download all (ZIP)", make_zip(files_map), "ica_separated.zip", "application/zip", key="dl_all_a")
    else:
        section_title("Input Mixtures"); audio_row(X, names, "mix_b")
        section_title("Recovered Components"); comp_names = [f"Component {i+1}" for i in range(len(out["S_est"]))]
        audio_row(out["S_est"], comp_names, "comp_b")
        st.caption("Order, sign and volume are arbitrary without ground truth references.")
        fig = plot_signal_grid([("Mixtures", X), ("ICA Components", out["S_est"])])
        st.pyplot(fig); plt.close(fig)
        files_map = {f"mixture_{n}.wav": X[i] for i in range(len(X))}
        files_map.update({f"component_{i+1}.wav": out["S_est"][i] for i in range(len(out["S_est"]))})
        st.download_button("Download all (ZIP)", make_zip(files_map), "ica_components.zip", "application/zip", key="dl_all_b")


def tab_pca(res):
    """Compare PCA decorrelation against FastICA higher-order independence."""
    if res["mode"] == "B":
        st.info("Needs clean sources (Mode A)"); return
    pca, names = res["pca"], res["names"]
    st.dataframe([{"Source": n, "PCA |corr|": round(pca["pca"]["abs_correlations"][i], 4),
                   "PCA SDR": round(pca["pca"]["sdr_db"][i], 2), "ICA |corr|": round(pca["ica"]["abs_correlations"][i], 4),
                   "ICA SDR": round(pca["ica"]["sdr_db"][i], 2)} for i, n in enumerate(names)])
    fig = plot_bars(names, pca["pca"], pca["ica"])
    st.pyplot(fig); plt.close(fig)
    audio_row(pca["pca"]["S_aligned"], names, "pca_rec")
    st.caption("PCA outputs: uncorrelated but still mixed.")
    st.markdown("PCA only diagonalizes the second-order covariance matrix, making components uncorrelated.\n"
                "Because whitened data has a spherical covariance, any rotation remains equally uncorrelated.\n"
                "FastICA leverages higher-order non-Gaussian statistics to identify the exact rotation for true independence.\n"
                "As seen above, PCA leaves audible cross-talk while ICA isolates clean sources.")


def tab_under_hood(res):
    """Examine covariance matrices, joint distributions, kurtosis, and iteration curves."""
    out = res["out"]
    section_title("a) Whitening Check")
    c1, c2 = st.columns(2)
    c1.caption("Covariance Before Whitening"); c1.dataframe(out["cov_before"].round(3))
    c2.caption("Covariance After Whitening"); c2.dataframe(out["cov_after"].round(3))
    st.caption("After whitening the covariance is approximately the identity matrix.")

    section_title("b) 2D Scatter Projections")
    panels = ([("Sources", res["S"])] if res["mode"] == "A" else []) + [
        ("Mixtures", res["X"]), ("After Whitening (Z)", out["Z"]), ("After ICA (S_est)", out["S_est"])
    ]
    fig_sc = plot_scatter(panels)
    st.pyplot(fig_sc); plt.close(fig_sc)
    st.caption("Whitening removes shearing and stretching into a round spherical cloud; ICA then finds the optimal rotation.")

    section_title("c) Kurtosis & Non-Gaussianity")
    groups = ({"Sources": res["S"]} if res["mode"] == "A" else {})
    groups.update({"Mixtures": res["X"], "Recovered": out["S_est"]})
    fig_k = plot_kurtosis(groups)
    st.pyplot(fig_k); plt.close(fig_k)
    st.caption("By the Central Limit Theorem, mixtures are closer to Gaussian (kurtosis nearer 0) than isolated sources.")

    section_title("d) FastICA Training Curves")
    fig_tr = plot_training(out["history"], res["curve"])
    st.pyplot(fig_tr); plt.close(fig_tr)


def tab_sweep(res):
    """Benchmark ICA separation degradation across various synthetic noise levels."""
    if res["mode"] == "B":
        st.info("Needs clean sources (Mode A)"); return
    if st.button("Run noise sweep (about 30 s)", key="btn_sweep"):
        with st.spinner("Sweeping SNR from 40 dB to 0 dB..."):
            st.session_state["sweep"] = snr_sweep(res["S"], res["A"])
    sweep = st.session_state.get("sweep")
    if sweep:
        st.dataframe([{"SNR (dB)": s[0], "Mean SDR (dB)": round(float(np.mean(s[1]["sdr_db"])), 2),
                       "Mean |corr|": round(float(np.mean(s[1]["abs_correlations"])), 4)} for s in sweep])
        fig = plot_sweep(sweep)
        st.pyplot(fig); plt.close(fig)
        match_10 = next((s[1] for s in sweep if s[0] == 10), None)
        if match_10:
            section_title("Recovered Audio at 10 dB Noise SNR")
            audio_row(match_10["S_aligned"], res["names"], "noisy10")


def tab_shazam(res):
    """Fingerprint audio library and identify query audio clips using peak hash offset voting."""
    lib_files = st.file_uploader("Song library: 5 to 10 audio files, any format", accept_multiple_files=True, type=AUDIO_TYPES, key="lib")
    if st.button("Build fingerprint database", key="btn_build_db"):
        if not lib_files:
            st.warning("Please upload library files first.")
        else:
            with st.spinner("Extracting spectral landmark hashes..."):
                songs = {os.path.splitext(f.name)[0]: convert_audio(f.getvalue(), f.name, 30) for f in lib_files}
                st.session_state["db"] = build_database(songs)
                st.session_state["songs"] = songs
                st.success(f"Indexed {len(songs)} songs with {len(st.session_state['db'])} unique hashes.")

    db = st.session_state.get("db")
    if db:
        q_mode = st.radio("Query", ["Random 5 s slice of a library song + noise", "A separated track from the ICA run", "Upload a query clip"], key="q_mode")
        clip, true_song = None, None
        if q_mode == "Random 5 s slice of a library song + noise":
            song_name = st.selectbox("Select target library song", list(st.session_state["songs"].keys()), key="sel_song")
            snr = st.slider("Noise SNR in dB", 0, 40, 10, key="shazam_snr")
            full_sig = st.session_state["songs"][song_name]
            dur_samples = min(len(full_sig), int(5.0 * SAMPLE_RATE))
            start = np.random.RandomState().randint(0, max(1, len(full_sig) - dur_samples))
            raw_slice = full_sig[start : start + dur_samples]
            clip = add_noise(raw_slice[np.newaxis, :], snr)[0]
            true_song = song_name
        elif q_mode == "A separated track from the ICA run":
            if not res:
                st.info("Run ICA separation first in the Separate tab."); return
            tracks = res["score"]["S_aligned"] if res["mode"] == "A" else res["out"]["S_est"]
            names = res["names"] if res["mode"] == "A" else [f"Component {i+1}" for i in range(len(tracks))]
            track_idx = st.selectbox("Select separated track", range(len(names)), format_func=lambda i: names[i], key="sel_rec")
            clip = tracks[track_idx]
            st.caption("For this test to succeed, ensure the original track exists in the fingerprint library.")
        else:
            q_file = st.file_uploader("Upload query clip", type=AUDIO_TYPES, key="query")
            if q_file:
                clip = convert_audio(q_file.getvalue(), q_file.name, 15)

        if clip is not None and st.button("Identify", key="btn_identify"):
            st.audio(to_wav_bytes(clip), format="audio/wav")
            best, scores = identify(db, clip)
            if best:
                msg = f"Best match: **{best}**"
                if true_song:
                    msg += "  (Correct!)" if best == true_song else f"  (Expected: {true_song})"
                st.success(msg)
            else:
                st.warning("No confident landmark match found.")
            fig = plot_scores(scores)
            st.pyplot(fig); plt.close(fig)


def tab_concepts():
    """Educational explanations covering foundational blind source separation concepts."""
    with st.expander("1. Why ICA over PCA"):
        st.write("PCA only analyzes second-order covariance, producing orthogonal axes that maximize variance. "
                 "However, zero correlation does not imply statistical independence. "
                 "Because whitened data possesses an isotropic spherical distribution, any orthogonal rotation preserves zero covariance. "
                 "ICA examines higher-order moments (kurtosis and negentropy) to isolate the single correct rotational orientation. "
                 "*Analogy:* PCA shapes pizza dough into a circular disc; ICA inspects the toppings to rotate the slices into position.")
    with st.expander("2. What whitening does and why it simplifies ICA"):
        st.write("Whitening standardizes signal variances to 1.0 and eliminates all cross-channel linear correlation. "
                 "Geometrically, it strips away shearing and scaling from mixing matrix A, converting an elongated ellipsoid into a unit sphere. "
                 "This guarantees that the remaining unknown transformation is strictly an orthogonal rotation matrix. "
                 "Instead of searching over n×n arbitrary matrix entries, ICA only has to solve for n(n-1)/2 rotation angles. "
                 "*Analogy:* Whitening inflates a crumpled ball into a round globe, leaving only the task of spinning it to face north.")
    with st.expander("3. Non-Gaussianity and the Central Limit Theorem"):
        st.write("The Central Limit Theorem establishes that sums of independent random variables are strictly more Gaussian than the individual terms. "
                 "Consequently, linear mixtures always exhibit higher Gaussianity than the original audio sources. "
                 "FastICA inverts this principle: it searches for projection vectors that maximize non-Gaussianity via the tanh contrast function. "
                 "A pure Gaussian source possesses isotropic symmetry with zero directional higher-order contrast. "
                 "Therefore, ICA can separate at most one Gaussian source; all other components must be non-Gaussian. "
                 "*Analogy:* Mixing colorful paints yields muddy gray; unmixing requires looking for the sharpest, least-gray primary colors.")
    with st.expander("4. Experiments, ambiguities, and metric selection"):
        st.write("Blind source separation has intrinsic ambiguities: neither original absolute volume, global sign, nor channel order can be recovered. "
                 "The match_and_score function handles this by normalizing variance, testing signed cross-correlations, and using Hungarian bipartite matching. "
                 "Signal-to-Distortion Ratio (SDR) in decibels measures reconstruction accuracy after optimal scalar projection. "
                 "The PCA-vs-ICA experiment demonstrates that decorrelation alone cannot unmix speech, while the SNR sweep measures noise robustness.")


def main():
    """Main Streamlit execution function."""
    st.set_page_config(page_title="Unmix the Signal", page_icon="🎧", layout="wide")
    apply_plot_theme()
    inject_css()
    hero()
    steps_row()

    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Mode", ["A: Clean sources (app mixes them)", "B: Real mixed recordings (app unmixes them)"], key="mode_rad")
    max_seconds = st.sidebar.slider("Max audio length (seconds)", 5, 20, 10, key="max_sec")
    seed, snr_db = 0, 60
    if "A:" in mode:
        seed = st.sidebar.slider("Mixing seed", 0, 100, 0, key="seed_sl")
        snr_db = st.sidebar.slider("Noise SNR in dB (60 = no noise)", 0, 60, 60, key="snr_sl")
    else:
        st.sidebar.caption("Mode B requires synchronized recordings without echo or room delay.")

    section_title("drop your audio")
    files = st.file_uploader("Drag and drop audio files here (2 to 4, any format)", type=AUDIO_TYPES, accept_multiple_files=True, key="main_files")
    if files and len(files) > 4:
        st.warning("Maximum 4 files allowed. Keeping the first 4 files."); files = files[:4]
    elif files and len(files) < 2:
        st.info("Upload at least 2 files.")

    if files and len(files) >= 2 and st.button("Run separation", type="primary", key="btn_run"):
        try:
            with st.spinner("Running FastICA separation..."):
                run_separation(files, mode, max_seconds, seed, snr_db)
        except Exception as e:
            st.error(f"Separation failed: {e}")

    res = st.session_state.get("res")
    tab_labels = ["Separate", "PCA vs ICA", "Under the hood", "Noise sweep", "Shazam", "Concepts"]
    t_sep, t_pca, t_hood, t_swp, t_shaz, t_conc = st.tabs(tab_labels)

    with t_sep:
        tab_separate(res) if res else st.info("Upload files and click Run separation")
    with t_pca:
        tab_pca(res) if res else st.info("Upload files and click Run separation")
    with t_hood:
        tab_under_hood(res) if res else st.info("Upload files and click Run separation")
    with t_swp:
        tab_sweep(res) if res else st.info("Upload files and click Run separation")
    with t_shaz:
        tab_shazam(res)
    with t_conc:
        tab_concepts()


if __name__ == "__main__":
    main()
