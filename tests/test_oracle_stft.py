"""STFT mit Gauß-Fenster und analytisches Signal gegen die Definition (direkte Summe) und gegen scipy (nur als Orakel)."""

import math

import numpy as np
import pytest
import scipy.signal as sps

import rea_signals as SG
import rea_stft as T
import signal_generator as G


def _direct_stft(z, window, t, m, nfft):
    """X(t, m) = sum_k z[t + k] w[k] exp(-i 2 pi m k / nfft), k = -K .. K, Werte außerhalb der Reihe null."""
    K = (len(window) - 1) // 2
    total = 0j
    for k in range(-K, K + 1):
        if 0 <= t + k < len(z):
            total += z[t + k] * window[k + K] * np.exp(-2j * np.pi * m * k / nfft)
    return total


def test_gauss_windows_match_the_definition_and_scipy():
    k, g, dg, tg = T.gauss_windows(5.0)
    assert k.tolist() == list(range(-20, 21)) and len(g) == 41
    assert g == pytest.approx(sps.windows.gaussian(41, std=5.0), abs=1e-14) and g[20] == 1.0
    assert tg == pytest.approx(k * g) and dg == pytest.approx(-k / 25.0 * g)
    # dg ist die Ableitung von g nach k (zentrale Differenz bis O(h^2))
    h = 1e-5
    num = (np.exp(-((k + h) ** 2) / 50.0) - np.exp(-((k - h) ** 2) / 50.0)) / (2 * h)
    assert dg == pytest.approx(num, abs=1e-8)


def test_window_length_is_four_sigma_rounded_up():
    assert len(T.gauss_windows(12.0)[0]) == 2 * 48 + 1 and len(T.gauss_windows(4.5)[0]) == 2 * 18 + 1


def test_stft3_equals_the_direct_sum_for_all_three_windows():
    rng = G.SplitMix64(3)
    z = rng.normal(40) + 1j * rng.normal(40)
    sigma, nfft = 2.0, 64
    k, g, dg, tg = T.gauss_windows(sigma)
    out = T.stft3(z, sigma, nfft)
    assert [o.shape for o in out] == [(40, 64)] * 3
    for X, w in zip(out, (g, dg, tg)):
        for t in (0, 1, 7, 20, 38, 39):
            for m in (0, 1, 5, 31, 32, 63):
                assert X[t, m] == pytest.approx(_direct_stft(z, w, t, m, nfft), abs=1e-10)


def test_stft_of_a_tone_peaks_at_the_tone_with_amplitude_sum_of_the_window():
    n, f0 = 256, 0.125
    z = np.exp(2j * np.pi * f0 * np.arange(n))
    X = T.stft3(z, 6.0, 256)[0]
    k, g, _dg, _tg = T.gauss_windows(6.0)
    assert np.argmax(np.abs(X[128])) == 32 and abs(X[128, 32]) == pytest.approx(g.sum(), rel=1e-12)
    assert T.spectrogram(z, 6.0, 256)[128, 32] == pytest.approx(g.sum() ** 2, rel=1e-12)


def test_shifting_the_input_shifts_the_spectrogram():
    rng = G.SplitMix64(5)
    z = rng.normal(200) + 1j * rng.normal(200)
    P = T.spectrogram(z, 3.0, 64)
    Q = T.spectrogram(np.roll(z, 17), 3.0, 64)
    assert Q[40:160] == pytest.approx(P[23:143], rel=1e-9, abs=1e-9)


def test_frequency_axis_and_too_small_fft():
    assert T.freq_axis(8).tolist() == [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    with pytest.raises(ValueError):
        T.stft3(np.zeros(20, complex), 10.0, 64)


def test_analytic_signal_against_scipy_hilbert():
    rng = G.SplitMix64(9)
    for n in (64, 65, 100):
        x = rng.normal(n)
        z = SG.analytic(x)
        assert z == pytest.approx(sps.hilbert(x), abs=1e-12) and z.real == pytest.approx(x, abs=1e-12)


def test_analytic_signal_has_no_negative_frequencies_and_is_exact_for_a_cosine():
    n = 128
    t = np.arange(n)
    x = np.cos(2 * np.pi * 10 * t / n + 0.7)
    z = SG.analytic(x)
    assert z == pytest.approx(np.exp(1j * (2 * np.pi * 10 * t / n + 0.7)), abs=1e-12)
    Z = np.fft.fft(SG.analytic(G.SplitMix64(2).normal(n)))
    assert np.abs(Z[n // 2 + 1:]).max() < 1e-12


def test_window_may_fill_the_fft_exactly_but_not_exceed_it():
    z = np.exp(2j * np.pi * 0.1 * np.arange(100))
    assert len(T.gauss_windows(8.0)[0]) == 65
    assert T.stft3(z, 8.0, 65)[0].shape == (100, 65)                      # Fensterlänge 65 passt genau in die FFT
    for nfft in (64, 63):
        with pytest.raises(ValueError):
            T.stft3(z, 8.0, nfft)
