#!/usr/bin/env python3
"""Print the moments where sounds start in a track (spectral-flux onsets).

    python3 tools/onsets.py assets/audio.m4a
    python3 tools/onsets.py my_music.mp3 --threshold 0.05

Use the printed times as `events` in scenario.py: every sound becomes a
bounce, an escape through a hole, or (with "any") whichever the physics
finds more natural.
"""

import argparse
import subprocess

import numpy as np


def load(path, sr=48000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768, sr


def onsets(x, sr, hop=256, n=2048, threshold=0.06, min_gap=0.07):
    win = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    mag = np.log1p(100 * np.abs(np.fft.rfft(frames * win, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(1)
    flux = np.concatenate([[0], flux])
    flux /= flux.max() + 1e-9
    k = 20
    med = np.array([np.median(flux[max(0, i - k):i + k]) for i in range(len(flux))])
    out, last = [], -1e9
    for i in range(1, len(flux) - 1):
        t = (i * hop + n / 2) / sr     # centre of the analysis window
        if (flux[i] > med[i] + threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]
                and t - last >= min_gap):
            out.append((t, float(flux[i])))
            last = t
    return [(snap(x, sr, t), st) for t, st in out]


def snap(x, sr, t, before=0.01, after=0.035):
    """Move a flux peak onto the attack: steepest rise of the loudness."""
    a = int(max(0, t - before) * sr)
    b = int(min(len(x), (t + after) * sr))
    w = int(0.002 * sr)
    seg = x[max(0, a - w):b] ** 2
    env = np.sqrt(np.convolve(seg, np.ones(w) / w, "valid") + 1e-10)
    rise = np.diff(np.log(env))
    rise = np.convolve(rise, np.ones(w // 2) / (w // 2), "same")
    return (max(0, a - w) + int(np.argmax(rise)) + w) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threshold", type=float, default=0.06)
    ap.add_argument("--min-gap", type=float, default=0.07)
    a = ap.parse_args()
    x, sr = load(a.audio)
    for t, s in onsets(x, sr, threshold=a.threshold, min_gap=a.min_gap):
        print(f"{t:7.3f}  strength {s:.2f}")


if __name__ == "__main__":
    main()
