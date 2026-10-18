"""Kurzzeit-Fourier-Transformation mit Gauß-Fenster, seiner Ableitung und seiner Zeitgewichtung (Grundlage von Neuzuordnung und Synchrosqueezing).

Fenster g[k] = exp(-k^2 / (2 sigma^2)), k = -K .. K, K = ceil(4 sigma); Ableitung dg[k] = -k / sigma^2 g[k], Zeitgewicht tg[k] = k g[k]. Der Rahmen t ist um den Zeitpunkt t zentriert, der Phasenbezug liegt am Rahmenzentrum.
"""

import math

import numpy as np

import rea_constants as C


def gauss_windows(sigma: float):
    """(k, g, dg, tg) mit k = -K .. K."""
    K = int(math.ceil(4.0 * sigma))
    k = np.arange(-K, K + 1)
    g = np.exp(-(k.astype(float) ** 2) / (2.0 * sigma ** 2))
    return k, g, -k / sigma ** 2 * g, k * g


def stft3(z, sigma: float, nfft: int = C.NFFT):
    """Drei STFTs (Fenster, Ableitungsfenster, Zeitgewichtsfenster) der Reihe z; jede Form (n, nfft), komplex, Frequenz m / nfft."""
    z = np.asarray(z, complex)
    n = len(z)
    k, g, dg, tg = gauss_windows(sigma)
    K = (len(k) - 1) // 2
    if 2 * K + 1 > nfft:
        raise ValueError("Fenster länger als die FFT")
    pad = np.concatenate([np.zeros(K, complex), z, np.zeros(K, complex)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, 2 * K + 1)
    out = []
    for w in (g, dg, tg):
        a = np.zeros((n, nfft), complex)
        a[:, :K + 1] = frames[:, K:] * w[K:]
        a[:, nfft - K:] = frames[:, :K] * w[:K]
        out.append(np.fft.fft(a, axis=1))
    return out


def spectrogram(z, sigma: float, nfft: int = C.NFFT) -> np.ndarray:
    """Leistung |STFT|^2, Form (n, nfft)."""
    return np.abs(stft3(z, sigma, nfft)[0]) ** 2


def freq_axis(nfft: int = C.NFFT) -> np.ndarray:
    return np.arange(nfft) / nfft
