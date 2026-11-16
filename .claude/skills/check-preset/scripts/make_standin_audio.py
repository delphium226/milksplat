"""Synthetic stand-in for the techno test recording.

Writes 32 s of 128 bpm techno as interleaved little-endian f32 stereo at
48 kHz: a quiet 11 s breakdown (pad, soft hats, filtered bass), then the drop
(kick, bass, hats, claps, stabs). Deterministic (fixed seed).

usage: python make_standin_audio.py [out_path]   (default .test-audio/techno.raw)
"""
import os
import sys

import numpy as np

SR = 48000
T = 32.0
N = int(SR * T)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
beat = 60.0 / 128.0
drop = 11.0


def env_at(times, length, decay):
    e = np.zeros(N)
    k = np.arange(int(length * SR)) / SR
    shape = np.exp(-k / decay)
    for s in times:
        i = int(s * SR)
        if i < N:
            j = min(i + len(shape), N)
            e[i:j] += shape[: j - i]
    return e


def lp(x, fc):
    """One-pole low-pass, run in blocks so it stays reasonably fast."""
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    acc = 0.0
    # Blocks short enough that a^-block stays well inside float64 precision.
    block = max(1, min(4096, int(25 / -np.log(a))))
    powers = a ** np.arange(1, block + 1)
    for s in range(0, len(x), block):
        xb = x[s : s + block] * (1 - a)
        n = len(xb)
        # y[k] = a^(k+1) acc + sum_{m<=k} a^(k-m) xb[m]
        w = a ** -np.arange(n)
        yb = powers[:n] * acc + np.cumsum(xb * w) / w
        y[s : s + n] = yb
        acc = yb[-1]
    return y


beats = np.arange(0, T, beat)
out = np.zeros(N)

# Kick: a pitch-dropping sine, after the drop (half-time in the last bars before it).
k = np.arange(int(0.35 * SR)) / SR
ks = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-k / 0.03)) / SR) * np.exp(-k / 0.12)
for s in beats:
    if s >= drop or (drop - 8 * beat < s < drop and int(round(s / beat)) % 2 == 0):
        i = int(s * SR)
        j = min(i + len(ks), N)
        out[i:j] += 0.9 * ks[: j - i]

# Bass: rolling offbeat saw, low-passed.
bass_env = env_at([s + beat / 2 for s in beats] + [s + 3 * beat / 4 for s in beats], 0.2, 0.06)
saw = 2 * ((t * 55.0) % 1.0) - 1
out += np.where(t >= drop, 0.6, 0.25) * lp(saw * bass_env, 300.0)

# Hats: closed 16ths and open offbeats, high-passed noise.
noise = rng.standard_normal(N)
hp = noise - lp(noise, 6000.0)
ch = env_at(np.arange(0, T, beat / 4), 0.05, 0.012)
oh = env_at([s + beat / 2 for s in beats], 0.25, 0.07)
out += hp * (0.18 * ch + np.where(t >= drop, 0.3, 0.12) * oh)

# Claps on 2 and 4 after the drop.
claps = [s for i, s in enumerate(beats) if i % 2 == 1 and s >= drop]
out += 1.2 * (lp(noise, 2500.0) - lp(noise, 900.0)) * env_at(claps, 0.2, 0.05)

# Stabs after the drop, a pad in the breakdown.
chord = [220.0, 261.63, 329.63]
stab_env = env_at([s + 2.5 * beat for s in np.arange(drop, T, 4 * beat)], 0.4, 0.12)
stab = sum(np.sign(np.sin(2 * np.pi * c * t)) for c in chord) / 3
out += 0.25 * lp(stab, 2000.0) * stab_env
pad = sum(np.sin(2 * np.pi * c * t + np.sin(t * 0.7)) for c in chord) / 3
out += np.where(t < drop, 0.12 + 0.08 * (t / drop), 0.04) * pad

out /= np.max(np.abs(out)) * 1.1
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(".test-audio", "techno.raw")
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
np.stack([out, out], axis=1).astype("<f4").tofile(path)
print(f"wrote {path} ({T:.0f} s stand-in)")
