"""Wigner-Ville und SPWVD gegen die Definition (Handrechnung, direkte Doppelsumme), Marginale und die geschlossene Form des Kreuzterms."""

import math

import numpy as np
import pytest
import scipy.signal as sps

import rea_signals as SG
import rea_wvd as V
import signal_generator as G


def _direct_wvd(z, M):
    """W(t, m) = sum_{|tau| <= M/2 - 1, t +- tau in der Reihe} z[t + tau] conj(z[t - tau]) exp(-i 2 pi m tau / M), Realteil."""
    n, L = len(z), M // 2 - 1
    W = np.zeros((n, M))
    for t in range(n):
        for m in range(M):
            total = 0j
            for tau in range(-L, L + 1):
                if 0 <= t + tau < n and 0 <= t - tau < n:
                    total += z[t + tau] * np.conj(z[t - tau]) * np.exp(-2j * np.pi * m * tau / M)
            W[t, m] = total.real
    return W


def test_mini_instance_by_hand(mini_x):
    R = V.lag_products(mini_x.astype(complex), 1)
    W = V.wvd_from_lags(R, 1, 8)
    by_hand = 4 + 6 * np.cos(2 * np.pi * np.arange(8) / 8)
    assert by_hand.round(4).tolist() == [10.0, 8.2426, 4.0, -0.2426, -2.0, -0.2426, 4.0, 8.2426]
    assert W[1] == pytest.approx(by_hand, abs=1e-12) and W[0] == pytest.approx(np.ones(8)) and W[2] == pytest.approx(9.0 * np.ones(8))


def test_lag_products_by_definition():
    rng = G.SplitMix64(4)
    z = rng.normal(12) + 1j * rng.normal(12)
    R = V.lag_products(z, 3)
    assert R.shape == (12, 7)
    for t in range(12):
        for tau in range(-3, 4):
            ok = 0 <= t + tau < 12 and 0 <= t - tau < 12
            assert R[t, tau + 3] == pytest.approx(z[t + tau] * np.conj(z[t - tau]) if ok else 0.0)


def test_wvd_equals_the_direct_double_sum_and_is_real():
    rng = G.SplitMix64(6)
    z = rng.normal(18) + 1j * rng.normal(18)
    M = 16
    assert V.wvd(z, M) == pytest.approx(_direct_wvd(z, M), abs=1e-10)
    R = V.lag_products(z, M // 2 - 1)
    full = np.zeros((18, M), complex)
    full[:, np.arange(-7, 8) % M] = R
    assert np.abs(np.fft.fft(full, axis=1).imag).max() < 1e-10          # R(t, -tau) = conj R(t, tau): die Summe ist reell


def test_frequency_marginal_is_M_times_the_instantaneous_power():
    rng = G.SplitMix64(8)
    z = rng.normal(30) + 1j * rng.normal(30)
    M = 32
    assert V.wvd(z, M).sum(axis=1) == pytest.approx(M * np.abs(z) ** 2, abs=1e-9)


def test_wvd_of_a_tone_is_a_line_and_of_a_chirp_follows_the_frequency():
    n = 512
    tone = SG.analytic(SG.tone(n, 96 / 1024).x)
    W = V.wvd(tone)
    assert (np.argmax(W[256:257], axis=1) == 96).all()
    sig = SG.linear_chirp(n, 0.05, 0.20)
    ridge = np.argmax(V.wvd(SG.analytic(sig.x)), axis=1) / 1024
    assert np.abs(ridge[100:412] - sig.tracks[0][100:412]).max() < 6e-4


def test_lag_product_of_two_tones_is_the_closed_form_with_a_cross_term():
    n, f1, f2, p1, p2, a1, a2 = 40, 0.11, 0.19, 0.3, 0.6, 1.0, 0.7
    t = np.arange(n)
    z1, z2 = a1 * np.exp(1j * (2 * np.pi * f1 * t + p1)), a2 * np.exp(1j * (2 * np.pi * f2 * t + p2))
    R = V.lag_products(z1 + z2, 10)
    for tt in range(10, 30):
        for tau in range(-10, 11):
            auto = a1 ** 2 * np.exp(1j * 4 * np.pi * f1 * tau) + a2 ** 2 * np.exp(1j * 4 * np.pi * f2 * tau)
            cross = 2 * a1 * a2 * math.cos(2 * np.pi * (f2 - f1) * tt + p2 - p1) * np.exp(1j * 2 * np.pi * (f1 + f2) * tau)
            assert R[tt, tau + 10] == pytest.approx(auto + cross, abs=1e-12)


def test_cross_term_peak_height_is_twice_the_tone_amplitude_product_times_the_lag_count():
    """Töne auf dem Raster (96/1024 und 192/1024): in der Mitte (144/1024) steht (2L + 1) 2 A1 A2 cos(2 pi (f2 - f1) t + p2 - p1) bis auf die Nebenkeulen der Töne (je höchstens 1 / sin(pi 0,09375) = 3,5)."""
    n, L = 512, 255
    sig = SG.two_tones(n, 96 / 1024, 192 / 1024)
    W = V.wvd(SG.analytic(sig.x))
    for t in (255, 256):
        expected = (2 * L + 1) * 2.0 * math.cos(2 * math.pi * (96 / 1024) * t + 0.3)
        assert W[t, 144] == pytest.approx(expected, abs=2 * 3.6)
        assert W[t, 96] > 0.9 * (2 * L + 1) - 10 and W[t, 192] > 0.9 * (2 * L + 1) - 10


def test_the_wvd_of_two_tones_has_negative_values_but_a_spectrogram_does_not():
    W = V.wvd(SG.analytic(SG.two_tones(512, 0.10, 0.18).x))
    assert W[100:412].min() < -0.9 * W[100:412].max() and W[100:412, 140:147].min() < 0


def test_smooth_time_equals_direct_convolution_and_keeps_constants():
    rng = G.SplitMix64(12)
    R = rng.normal(70)[:, None] * np.ones((1, 5)) + 1j * rng.normal(70)[:, None]
    for st in (1.0, 3.5):
        K = int(math.ceil(4 * st))
        u = np.arange(-K, K + 1)
        g = np.exp(-(u.astype(float) ** 2) / (2 * st ** 2))
        g /= g.sum()
        expected = np.stack([np.convolve(R[:, j], g, mode="same") for j in range(5)], axis=1)
        assert V.smooth_time(R, st) == pytest.approx(expected, abs=1e-12)
    const = np.ones((100, 3), complex)
    assert V.smooth_time(const, 4.0)[30:70] == pytest.approx(const[30:70], abs=1e-12) and V.smooth_time(const, 0.0) is const


def test_spwvd_without_smoothing_and_with_a_huge_lag_window_is_the_wvd():
    z = SG.analytic(SG.linear_chirp(128, 0.05, 0.2).x)
    assert V.spwvd(z, 0.0, 1e9, 64) == pytest.approx(V.wvd(z, 64), abs=1e-9)


def test_time_smoothing_removes_the_cross_term_that_lag_windowing_alone_keeps():
    z = SG.analytic(SG.two_tones(512, 0.10, 0.18).x)
    lo, hi, mid, tone = 100, 412, slice(140, 147), 102
    peak = V.wvd(z)[lo:hi, tone - 3:tone + 4].max(axis=1).mean()

    def ratio(W):
        return np.abs(W[lo:hi, mid]).max(axis=1).mean() / np.abs(W[lo:hi, tone - 3:tone + 4]).max(axis=1).mean()
    assert ratio(V.wvd(z)) > 1.0 and ratio(V.spwvd(z, 0.0, 20.0)) > 1.0 and ratio(V.spwvd(z, 6.0, 20.0)) < 0.05 and ratio(V.spwvd(z, 12.0, 20.0)) < 1e-3 and peak > 0


def test_wvd_matrix_size_guard():
    with pytest.raises(ValueError):
        V.wvd_from_lags(np.zeros((4, 9), complex), 4, 8)


def test_lag_count_may_fill_the_fft_exactly():
    W = V.wvd_from_lags(np.ones((4, 7), complex), 3, 7)                    # 2 * 3 + 1 = 7 = M
    assert W.shape == (4, 7) and W[0, 0] == pytest.approx(7.0) and W[0, 1:] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        V.wvd_from_lags(np.ones((4, 7), complex), 3, 6)
