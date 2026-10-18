"""Reassignment (Neuzuordnung) und Synchrosqueezing auf der Gauß-STFT.

Der Energieschwerpunkt des Punktes (t, f) liegt bei t_hat = t + Re(X_tg / X_g) und f_hat = f - Im(X_dg / X_g) / (2 pi) (Kodera et al. 1976, Auger und Flandrin 1995, hier mit Zeitgewichts- und Ableitungsfenster).
Reassignment verschiebt den Wert nach (t_hat, f_hat); Synchrosqueezing verschiebt nur in der Frequenz nach f_hat und sammelt die komplexen Werte.
"""

import math

import numpy as np

import rea_constants as C
import rea_stft as T

TWO_PI = 2.0 * math.pi


def reassign_fields(z, sigma: float, nfft: int = C.NFFT, floor_db: float = C.DEFAULT_FLOOR, abs_floor: float = 0.0):
    """(X, f_hat, t_hat, Maske): STFT, neu zugeordnete Frequenz (Zyklen je Abtastwert) und Zeit je Punkt; Maske: Leistung über floor_db gegen die Spitze und Betrag über abs_floor."""
    X, Xd, Xt = T.stft3(z, sigma, nfft)
    w = TWO_PI * T.freq_axis(nfft)[None, :]
    safe = np.where(np.abs(X) > 0, X, 1.0)
    f_hat = (w - np.imag(Xd / safe)) / TWO_PI
    t_hat = np.arange(len(z))[:, None] + np.real(Xt / safe)
    p = np.abs(X) ** 2
    mask = (p > 10.0 ** (floor_db / 10.0) * p.max()) & (np.abs(X) > abs_floor)
    return X, f_hat, t_hat, mask


def _accumulate(rows, cols, weights, shape):
    out = np.bincount(rows * shape[1] + cols, weights=weights, minlength=shape[0] * shape[1])
    return out.reshape(shape)


def reassigned_from(X, t_hat, f_hat, mask, nfft: int = C.NFFT) -> np.ndarray:
    """Neu zugeordnetes Spektrogramm aus den Feldern von reassign_fields: die Leistung jedes Punktes wandert auf den nächsten Rasterpunkt von (t_hat, f_hat); Form (n, nfft)."""
    n = X.shape[0]
    ti = np.rint(t_hat).astype(int)
    fi = np.rint(f_hat * nfft).astype(int)
    ok = mask & (ti >= 0) & (ti < n) & (fi >= 0) & (fi < nfft)
    return _accumulate(ti[ok], fi[ok], (np.abs(X) ** 2)[ok], (n, nfft))


def squeezed_from(X, f_hat, mask, nfft: int = C.NFFT) -> np.ndarray:
    """Synchrosqueezing (nur in der Frequenz) aus den Feldern: die komplexen STFT-Werte werden auf dem Rasterpunkt der neu zugeordneten Frequenz gesammelt; Form (n, nfft), komplex."""
    n = X.shape[0]
    fi = np.rint(f_hat * nfft).astype(int)
    ok = mask & (fi >= 0) & (fi < nfft)
    rows = np.broadcast_to(np.arange(n)[:, None], fi.shape)
    r, c, v = rows[ok], fi[ok], X[ok]
    return _accumulate(r, c, v.real, (n, nfft)) + 1j * _accumulate(r, c, v.imag, (n, nfft))


def reassigned_spectrogram(z, sigma: float, nfft: int = C.NFFT, floor_db: float = C.DEFAULT_FLOOR, abs_floor: float = 0.0) -> np.ndarray:
    X, f_hat, t_hat, mask = reassign_fields(z, sigma, nfft, floor_db, abs_floor)
    return reassigned_from(X, t_hat, f_hat, mask, nfft)


def synchrosqueeze(z, sigma: float, nfft: int = C.NFFT, floor_db: float = C.DEFAULT_FLOOR, abs_floor: float = 0.0) -> np.ndarray:
    X, f_hat, _t, mask = reassign_fields(z, sigma, nfft, floor_db, abs_floor)
    return squeezed_from(X, f_hat, mask, nfft)


def noise_floor(sigma: float, noise_sd: float, factor: float = C.GATE_FACTOR) -> float:
    """Betragsschwelle für die STFT: factor mal der Effektivwert von |X| bei analytischem weißem Rauschen der reellen Standardabweichung noise_sd, sqrt(E|X|^2) = 2 noise_sd sqrt(sum g^2)
    (die Analytik verdoppelt die Amplitude im positiven Band)."""
    _k, g, _dg, _tg = T.gauss_windows(sigma)
    return factor * 2.0 * noise_sd * math.sqrt(float(np.sum(g ** 2)))
