"""Kennzahlen und Bilder der fünf Verfahren: wirksame Breite, Kamm und sein Fehler, Gipfelzahl, Anteil auf den wahren Linien, Kreuzterm und negative Werte.

Alle Bilder sind quadratische Größen der Form (n, NFFT/2) auf dem Raster f = m / NFFT (Zyklen je Abtastwert, bis 0,5): Spektrogramm |X|^2, neu zugeordnetes Spektrogramm, Synchrosqueezing |S|^2,
Wigner-Ville und SPWVD (diese können negativ sein). So sind Breiten und Kreuztermverhältnisse zwischen den Verfahren vergleichbar.
"""

import math

import numpy as np

import rea_constants as C
import rea_reassign as R
import rea_stft as T
import rea_wvd as V

H = C.NFFT // 2                                    # Spaltenzahl der Bilder (Frequenz unter 0,5)


def inner(n: int) -> tuple[int, int]:
    """Zeitfenster der Breiten und Verhältnisse: ohne die Ränder, an denen das Fenster abgeschnitten wird (100 bis 411 bei n = 512)."""
    lo = int(round(n * 100 / 512))
    return lo, n - lo


def ridge_window(n: int) -> tuple[int, int]:
    """Zeitfenster der Frequenzfehler (60 bis 451 bei n = 512)."""
    lo = int(round(n * 60 / 512))
    return lo, n - lo


def effective_width(img, lo: int, hi: int) -> float:
    """Wirksame Breite je Zeit in Rasterpunkten: Summe der Beträge gegen das Maximum, gemittelt über die Zeiten lo .. hi - 1 (1 = eine scharfe Linie, ein Rasterpunkt)."""
    a = np.abs(np.asarray(img)[lo:hi, :H])
    mx = a.max(axis=1)
    ok = mx > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(a[ok].sum(axis=1) / mx[ok]))


def time_profile(img, lo: int, hi: int) -> np.ndarray:
    """Summe der Beträge über die Zeiten lo .. hi - 1, je Frequenzpunkt."""
    return np.abs(np.asarray(img)[lo:hi, :H]).sum(axis=0)


def count_peaks(profile, level: float = C.PEAK_LEVEL) -> int:
    """Zahl der lokalen Maxima über level mal dem höchsten (links strikt, rechts nicht strikt größer)."""
    p = np.asarray(profile, float)
    if len(p) < 3 or p.max() <= 0:
        return 0
    inner_ = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > level * p.max())
    return int(inner_.sum())


def band_share(img, tracks, lo: int, hi: int, band: int = C.BAND) -> float:
    """Anteil der Beträge (über die Zeiten lo .. hi - 1), der innerhalb von ± band Rasterpunkten einer wahren Linie liegt (zeitabhängig bei Chirps)."""
    a = np.abs(np.asarray(img)[lo:hi, :H])
    cols = np.arange(H)[None, :]
    near = np.zeros(a.shape, bool)
    for tr in tracks:
        centre = np.rint(np.asarray(tr)[lo:hi] * C.NFFT).astype(int)[:, None]
        near |= np.abs(cols - centre) <= band
    total = a.sum()
    return float(a[near].sum() / total) if total > 0 else float("nan")


def cross_ratio(img, f1: float, f2: float, lo: int, hi: int, band: int = C.BAND) -> float:
    """Betrag in der Mitte zwischen zwei Tönen gegen den am ersten Ton (jeweils größter Wert im Band ± band, gemittelt über die Zeiten lo .. hi - 1)."""
    a = np.abs(np.asarray(img)[lo:hi, :H])
    mid = int(0.5 * (f1 + f2) * C.NFFT)
    c1 = int(f1 * C.NFFT)
    den = float(a[:, c1 - band:c1 + band + 1].max(axis=1).mean())
    return float(a[:, mid - band:mid + band + 1].max(axis=1).mean() / den) if den > 0 else float("nan")


def negative_depth(img, lo: int, hi: int) -> float:
    """Kleinster Wert gegen den größten (im Fenster lo .. hi - 1): null für Verteilungen ohne negative Werte, bei Wigner-Ville negativ."""
    a = np.asarray(img)[lo:hi, :H]
    top = float(a.max())
    return float(a.min() / top) if top > 0 else 0.0


def cross_negative(img, f1: float, f2: float, lo: int, hi: int, band: int = C.BAND) -> float:
    """Kleinster Wert im Band um die Mitte zwischen zwei Tönen gegen den größten Betrag im Band um den ersten Ton (Zeiten lo .. hi - 1); bei Wigner-Ville negativ."""
    a = np.asarray(img)[lo:hi, :H]
    mid = int(0.5 * (f1 + f2) * C.NFFT)
    c1 = int(f1 * C.NFFT)
    den = float(np.abs(a[:, c1 - band:c1 + band + 1]).max())
    return float(a[:, mid - band:mid + band + 1].min() / den) if den > 0 else float("nan")


def rmse(est, truth, lo: int, hi: int) -> float:
    d = np.asarray(est, float)[lo:hi] - np.asarray(truth, float)[lo:hi]
    return float(math.sqrt(np.mean(d * d)))


# --- Bilder und Kämme -------------------------------------------------------------------------------------------------------------------------


def peak_estimates(z, sigma: float, abs_floor: float | None = None, sigma_t: float = C.DEFAULT_SIGMA_T, sigma_lag: float = C.DEFAULT_SIGMA_LAG, with_wvd: bool = True) -> dict:
    """Frequenzschätzungen je Zeitpunkt aus dem Gipfel jedes Verfahrens (Zyklen je Abtastwert):
    'Spektrogramm' (Gipfel der STFT), 'Neu zugeordnet' (neu zugeordnete Frequenz am Gipfel), 'Neu zugeordnet, Zeit' (neu zugeordnete Zeit zu diesem Gipfel), 'Synchrosqueezing' (Gipfel),
    'Synchrosqueezing mit Schwelle' (falls abs_floor), 'Wigner-Ville' und 'SPWVD' (Gipfel; nur mit with_wvd)."""
    X, f_hat, t_hat, _ = R.reassign_fields(z, sigma, C.NFFT, -300.0)
    rows = np.arange(len(z))
    idx = np.argmax(np.abs(X[:, :H]), axis=1)
    out = {"Spektrogramm": idx / C.NFFT, "Neu zugeordnet": f_hat[rows, idx], "Neu zugeordnet, Zeit": t_hat[rows, idx]}
    full = np.ones(X.shape, bool)
    out["Synchrosqueezing"] = np.argmax(np.abs(R.squeezed_from(X, f_hat, full)[:, :H]), axis=1) / C.NFFT
    if abs_floor is not None:
        out["Synchrosqueezing mit Schwelle"] = np.argmax(np.abs(R.squeezed_from(X, f_hat, np.abs(X) > abs_floor)[:, :H]), axis=1) / C.NFFT
    if with_wvd:
        lags = V.lag_products(z, 255)
        out["Wigner-Ville"] = np.argmax(V.wvd_from_lags(lags, 255, 512), axis=1) / C.NFFT
        taus = np.arange(-255, 256)
        smooth = V.smooth_time(lags * np.exp(-(taus.astype(float) ** 2) / (2.0 * sigma_lag ** 2))[None, :], sigma_t)
        out["SPWVD"] = np.argmax(V.wvd_from_lags(smooth, 255, 512), axis=1) / C.NFFT
    return out


def ridge_errors(est: dict, truth, n: int) -> dict:
    """RMSE jeder Schätzung gegen die wahre Frequenz (Zeiten ridge_window); 'Neu zugeordnet, Zeit' wird nicht bewertet, sondern liefert 'Neu zugeordnet (zur neuen Zeit)': die neu zugeordnete Frequenz
    gegen die wahre Frequenz zur neu zugeordneten Zeit."""
    lo, hi = ridge_window(n)
    truth = np.asarray(truth, float)
    out = {k: rmse(v, truth, lo, hi) for k, v in est.items() if k != "Neu zugeordnet, Zeit"}
    tt = est["Neu zugeordnet, Zeit"][lo:hi]
    ok = (tt >= 0) & (tt <= n - 1)
    ref = np.interp(tt[ok], np.arange(n), truth)
    d = est["Neu zugeordnet"][lo:hi][ok] - ref
    out["Neu zugeordnet (zur neuen Zeit)"] = float(math.sqrt(np.mean(d * d)))
    return out


def method_image(z, method: str, sigma: float, sigma_t: float = C.DEFAULT_SIGMA_T, sigma_lag: float = C.DEFAULT_SIGMA_LAG, floor_db: float = C.DEFAULT_FLOOR, abs_floor: float = 0.0):
    """(Bild, Kamm): Bild der Form (n, NFFT/2) des Verfahrens (quadratische Größe) und der Kamm (Gipfel je Zeitpunkt, Zyklen je Abtastwert)."""
    if method == "Spektrogramm":
        img = T.spectrogram(z, sigma)[:, :H]
        return img, np.argmax(img, axis=1) / C.NFFT
    if method == "Neu zugeordnet":
        X, f_hat, t_hat, mask = R.reassign_fields(z, sigma, C.NFFT, floor_db, abs_floor)
        idx = np.argmax(np.abs(X[:, :H]), axis=1)
        return R.reassigned_from(X, t_hat, f_hat, mask)[:, :H], f_hat[np.arange(len(z)), idx]
    if method == "Synchrosqueezing":
        S = R.synchrosqueeze(z, sigma, C.NFFT, floor_db, abs_floor)[:, :H]
        return np.abs(S) ** 2, np.argmax(np.abs(S), axis=1) / C.NFFT
    if method == "Wigner-Ville":
        W = V.wvd(z)
        return W, np.argmax(W, axis=1) / C.NFFT
    if method == "SPWVD":
        W = V.spwvd(z, sigma_t, sigma_lag)
        return W, np.argmax(W, axis=1) / C.NFFT
    raise ValueError(f"unbekanntes Verfahren: {method}")
