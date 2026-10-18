"""Testsignale (Ton, Chirp, Sinus-FM, zwei Töne, zwei parallele Chirps) als reelle Reihen mit wahrer Momentanfrequenz, analytisches Signal und Rauschen.

Frequenzen in Zyklen je Abtastwert (Abtastrate 1); alle Komponenten haben die Amplitude 1 und bleiben unter 0,25, damit die Wigner-Ville-Verteilung des analytischen Signals nicht faltet.
"""

import math
from dataclasses import dataclass

import numpy as np

import rea_constants as C

TWO_PI = 2.0 * math.pi


def analytic(x) -> np.ndarray:
    """Analytisches Signal über die FFT (nur positive Frequenzen, doppelt gewichtet); der Realteil ist die Eingabe."""
    x = np.asarray(x, float)
    n = len(x)
    h = np.zeros(n)
    h[0] = 1.0
    h[1:(n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.fft.ifft(np.fft.fft(x) * h)


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung reellen weißen Rauschens, dessen Leistung gegen die einer Linie der Amplitude 1 (Leistung 1/2) das Verhältnis snr_db hat."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class TestSignal:
    __test__ = False                # kein pytest-Testfall

    x: np.ndarray                   # reelle Reihe
    tracks: tuple                   # wahre Momentanfrequenz je Komponente (Arrays der Länge n)
    name: str


def tone(n: int, f: float, phase: float = 0.3) -> TestSignal:
    t = np.arange(n)
    return TestSignal(np.cos(TWO_PI * f * t + phase), (np.full(n, float(f)),), "Ton")


def linear_chirp(n: int, f0: float, f1: float, phase: float = 0.0) -> TestSignal:
    """Frequenz linear von f0 (t = 0) bis f1 (t = n - 1); Phase 2 pi (f0 t + (f1 - f0) t^2 / (2 (n - 1)))."""
    t = np.arange(n)
    ph = TWO_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * (n - 1))) + phase
    return TestSignal(np.cos(ph), (f0 + (f1 - f0) * t / (n - 1),), "Chirp")


def sine_fm(n: int, fc: float, df: float, fm: float) -> TestSignal:
    """Frequenz fc + df sin(2 pi fm t): Phase 2 pi fc t - (df / fm) cos(2 pi fm t)."""
    t = np.arange(n)
    ph = TWO_PI * fc * t - (df / fm) * np.cos(TWO_PI * fm * t)
    return TestSignal(np.cos(ph), (fc + df * np.sin(TWO_PI * fm * t),), "Sinus-FM")


def two_tones(n: int, f1: float, f2: float) -> TestSignal:
    t = np.arange(n)
    x = np.cos(TWO_PI * f1 * t + 0.3) + np.cos(TWO_PI * f2 * t + 0.6)
    return TestSignal(x, (np.full(n, float(f1)), np.full(n, float(f2))), "Zwei Töne")


def parallel_chirps(n: int, f0: float, f1: float, df: float) -> TestSignal:
    """Zwei lineare Chirps im festen Abstand df (der zweite mit Phase 0,6)."""
    a = linear_chirp(n, f0, f1, 0.3)
    b = linear_chirp(n, f0 + df, f1 + df, 0.6)
    return TestSignal(a.x + b.x, (a.tracks[0], b.tracks[0]), "Zwei parallele Chirps")


def make_signal(kind: str, n: int, df: float = C.DEFAULT_DF) -> TestSignal:
    if kind == "Ton":
        return tone(n, C.TONE_F)
    if kind == "Chirp":
        return linear_chirp(n, C.CHIRP_F0, C.CHIRP_F1)
    if kind == "Sinus-FM":
        return sine_fm(n, C.SINFM_FC, C.SINFM_DF, C.SINFM_FM)
    if kind == "Zwei Töne":
        return two_tones(n, C.TWO_F1, C.TWO_F1 + df)
    if kind == "Zwei parallele Chirps":
        return parallel_chirps(n, C.PAR_F0, C.PAR_F1, df)
    raise ValueError(f"unbekanntes Signal: {kind}")
