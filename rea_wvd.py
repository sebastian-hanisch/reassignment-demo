"""Wigner-Ville-Verteilung und geglättete Pseudo-Wigner-Ville-Verteilung (SPWVD) des analytischen Signals.

W(t, f) = sum_tau z[t + tau] conj(z[t - tau]) exp(-i 4 pi f tau) auf dem Raster f = m / (2 M), m = 0 .. M - 1 (also m / 1024 bei M = 512); Verzögerungen |tau| <= M/2 - 1, außerhalb der Reihe null.
Die SPWVD gewichtet die Verzögerung mit einem Gauß-Fenster (sigma_lag) und glättet die Verzögerungsprodukte in der Zeit mit einem Gauß-Fenster (sigma_t): Cohen-Klasse mit getrenntem Kern.
"""

import math

import numpy as np


def lag_products(z, max_lag: int) -> np.ndarray:
    """R[t, tau + max_lag] = z[t + tau] conj(z[t - tau]) für |tau| <= max_lag, null, wo t + tau oder t - tau außerhalb der Reihe liegt; Form (n, 2 max_lag + 1)."""
    z = np.asarray(z, complex)
    n = len(z)
    t = np.arange(n)[:, None]
    tau = np.arange(-max_lag, max_lag + 1)[None, :]
    a, b = t + tau, t - tau
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.where(ok, z[np.clip(a, 0, n - 1)] * np.conj(z[np.clip(b, 0, n - 1)]), 0.0)


def wvd_from_lags(R, max_lag: int, M: int) -> np.ndarray:
    """Spektrum über die Verzögerung: W(t, m) = Re sum_tau R(t, tau) exp(-i 2 pi m tau / M), m = 0 .. M - 1; Form (n, M) reell."""
    if 2 * max_lag + 1 > M:
        raise ValueError("M muss mindestens 2 max_lag + 1 sein")
    a = np.zeros((R.shape[0], M), complex)
    a[:, np.arange(-max_lag, max_lag + 1) % M] = R
    return np.real(np.fft.fft(a, axis=1))


def wvd(z, M: int = 512) -> np.ndarray:
    max_lag = M // 2 - 1
    return wvd_from_lags(lag_products(z, max_lag), max_lag, M)


def smooth_time(R, sigma_t: float) -> np.ndarray:
    """Gauß-Glättung (Summe 1) entlang der Zeit (Achse 0) mit Nullen am Rand, per FFT-Faltung."""
    if sigma_t <= 0:
        return R
    K = int(math.ceil(4.0 * sigma_t))
    u = np.arange(-K, K + 1)
    g = np.exp(-(u.astype(float) ** 2) / (2.0 * sigma_t ** 2))
    g /= g.sum()
    n = R.shape[0]
    size = 1 << int(math.ceil(math.log2(n + 2 * K)))
    full = np.fft.ifft(np.fft.fft(R, size, axis=0) * np.fft.fft(g, size)[:, None], axis=0)
    return full[K:K + n]


def spwvd(z, sigma_t: float, sigma_lag: float, M: int = 512) -> np.ndarray:
    max_lag = M // 2 - 1
    R = lag_products(z, max_lag)
    taus = np.arange(-max_lag, max_lag + 1)
    R = R * np.exp(-(taus.astype(float) ** 2) / (2.0 * sigma_lag ** 2))[None, :]
    return wvd_from_lags(smooth_time(R, sigma_t), max_lag, M)
