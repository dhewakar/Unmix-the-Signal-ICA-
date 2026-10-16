"""Audio fingerprinting and recognition module using combinatorial spectral peak hashing."""

from collections import defaultdict
import numpy as np

BANDS = [(5, 40), (40, 80), (80, 160), (160, 320)]


# Extract peak hashes and anchor timestamps from a 1D audio signal
def fingerprint(signal):
    """Normalise signal, compute STFT spectrogram, pick peaks, and create landmark hashes."""
    # signal shape: (n_samples,)
    max_val = np.max(np.abs(signal))
    norm_sig = signal / max_val if max_val > 0 else signal

    frame_size, hop = 2048, 512
    window = np.hanning(frame_size)
    num_frames = (len(norm_sig) - frame_size) // hop
    peaks = []

    for frame_idx in range(max(0, num_frames)):
        frame = norm_sig[frame_idx * hop : frame_idx * hop + frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        threshold = np.mean(spectrum) * 1.5

        for low, high in BANDS:
            band_slice = spectrum[low:high]
            if len(band_slice) > 0 and np.max(band_slice) > threshold:
                best_bin = low + int(np.argmax(band_slice))
                peaks.append((frame_idx, best_bin))

    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        count = 0
        for t2, f2 in peaks[i + 1 :]:
            if t2 - t1 > 64:
                break
            if t2 > t1:
                hashes.append(((f1, f2, t2 - t1), t1))
                count += 1
                if count == 3:
                    break
    return hashes


# Build an inverted index mapping spectral hashes to song names and timestamps
def build_database(songs):
    """Create an inverted hash table mapping each hash to (song_name, anchor_time)."""
    db = defaultdict(list)
    for song_name, signal in songs.items():
        for h, t in fingerprint(signal):
            db[h].append((song_name, t))
    return dict(db)


# Match query clip against database using temporal alignment offset voting
def identify(database, clip):
    """Find the best-matching song by identifying the highest single-offset vote count."""
    clip_hashes = fingerprint(clip)
    votes = defaultdict(lambda: defaultdict(int))

    for h, clip_time in clip_hashes:
        for song_name, song_time in database.get(h, []):
            # A true match lines up consistently in time across multiple landmarks
            offset = song_time - clip_time
            votes[song_name][offset] += 1

    scores = {}
    for song_name in votes:
        scores[song_name] = max(votes[song_name].values()) if votes[song_name] else 0

    best_name = None
    if scores and max(scores.values()) > 0:
        best_name = max(scores, key=scores.get)
    return best_name, scores


if __name__ == "__main__":
    np.random.seed(0)
    sample_rate = 22050
    duration = 15.0
    chunk_dur = 0.25
    chunk_samples = int(sample_rate * chunk_dur)
    num_chunks = int(duration / chunk_dur)

    # Synthesize 3 random sine-tone melody songs
    songs = {}
    for song_idx in range(1, 4):
        track = []
        for _ in range(num_chunks):
            freq = np.random.uniform(200.0, 2000.0)
            t_chunk = np.linspace(0, chunk_dur, chunk_samples, endpoint=False)
            track.append(np.sin(2 * np.pi * freq * t_chunk))
        songs[f"Song_{song_idx}"] = np.concatenate(track)

    db = build_database(songs)

    # Cut a 5-second slice from Song 2 and add 10 dB Gaussian noise
    clip_samples = int(5.0 * sample_rate)
    start_idx = int(3.0 * sample_rate)
    clean_clip = songs["Song_2"][start_idx : start_idx + clip_samples]

    signal_power = np.mean(clean_clip**2)
    noise_power = signal_power / (10.0 ** (10.0 / 10.0))
    noisy_clip = clean_clip + np.random.normal(0, np.sqrt(noise_power), clean_clip.shape)

    best_song, all_scores = identify(db, noisy_clip)
    print("Scores per song:", all_scores)
    print("Identified Song:", best_song)
