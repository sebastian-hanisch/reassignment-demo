"""Kennzahlen gegen handgebaute Bilder: wirksame Breite, Gipfelzahl, Anteil auf den Linien, Kreuzterm, negative Werte, Kämme und Fehler."""

import numpy as np
import pytest

import rea_constants as C
import rea_measure as M
import rea_signals as SG
import signal_generator as G

H = M.H


def _img(rows=10):
    return np.zeros((rows, H))


def test_time_windows():
    assert M.inner(512) == (100, 412) and M.ridge_window(512) == (60, 452) and M.inner(256) == (50, 206) and M.ridge_window(1024) == (120, 904) and H == 512


def test_effective_width_by_hand():
    a = _img()
    a[:, 7] = 4.0
    assert M.effective_width(a, 0, 10) == 1.0                                  # eine scharfe Linie
    b = _img()
    b[:, 20:25] = 3.0
    assert M.effective_width(b, 0, 10) == 5.0                                  # fünf gleich hohe Punkte
    c = _img()
    c[:, 3], c[:, 4], c[:, 5] = 1.0, 0.5, 0.5
    assert M.effective_width(c, 2, 8) == 2.0                                   # (1 + 0,5 + 0,5) / 1
    d = _img()
    d[:5, 3] = 2.0                                                             # Zeilen 5 bis 9 sind leer und zählen nicht
    assert M.effective_width(d, 0, 10) == 1.0 and np.isnan(M.effective_width(_img(), 0, 10))
    e = _img()
    e[:, 1], e[:, 2] = -3.0, 3.0                                               # Beträge: negative Werte zählen mit
    assert M.effective_width(e, 0, 10) == 2.0
    f = _img()
    f[:, H - 1] = 1.0
    assert M.effective_width(f, 0, 10) == 1.0


def test_effective_width_uses_only_the_requested_times():
    a = _img()
    a[:5, 3] = 1.0
    a[5:, 3:8] = 1.0
    assert M.effective_width(a, 0, 5) == 1.0 and M.effective_width(a, 5, 10) == 5.0 and M.effective_width(a, 0, 10) == 3.0


def test_count_peaks_by_hand():
    assert M.count_peaks([0, 1, 0, 0.5, 0, 0.1, 0]) == 2                       # 0,1 liegt unter 20 % von 1
    assert M.count_peaks([0, 1, 0, 0.2, 0]) == 1                               # genau 20 % zählt nicht (strikt darüber)
    assert M.count_peaks([0, 1, 0, 0.21, 0]) == 2
    assert M.count_peaks([0, 1, 1, 0]) == 1                                    # Plateau: ein Gipfel (links strikt, rechts nicht strikt)
    assert M.count_peaks([1, 0, 1]) == 0 and M.count_peaks([0, 0, 0]) == 0 and M.count_peaks([1, 2]) == 0
    assert M.count_peaks([0, 3, 0, 2, 0], level=0.7) == 0 + 1 + 0                # 2 liegt unter 0,7 * 3 = 2,1
    assert M.count_peaks([0, 3, 0, 2.2, 0], level=0.7) == 2


def test_time_profile_sums_the_magnitudes_over_the_window():
    a = _img()
    a[2:6, 9] = 2.0
    a[6:8, 9] = -1.0
    p = M.time_profile(a, 2, 8)
    assert p.shape == (H,) and p[9] == 4 * 2 + 2 * 1 and p.sum() == p[9] and M.time_profile(a, 0, 2).sum() == 0


def test_band_share_by_hand():
    a = _img()
    a[:, 100] = 2.0                                                            # genau auf der Linie
    a[:, 103] = 1.0                                                            # am Rand des Bandes (± 3)
    a[:, 104] = 1.0                                                            # knapp außerhalb
    track = [np.full(10, 100 / C.NFFT)]
    assert M.band_share(a, track, 0, 10) == pytest.approx(3.0 / 4.0)
    assert M.band_share(a, track, 0, 10, band=4) == pytest.approx(1.0) and M.band_share(a, track, 0, 10, band=0) == pytest.approx(0.5)
    two = [np.full(10, 100 / C.NFFT), np.full(10, 104 / C.NFFT)]
    assert M.band_share(a, two, 0, 10) == pytest.approx(1.0)
    ramp = _img()
    for t in range(10):
        ramp[t, 50 + t] = 1.0
    assert M.band_share(ramp, [(50 + np.arange(10)) / C.NFFT], 0, 10) == 1.0 and M.band_share(ramp, [np.full(10, 50 / C.NFFT)], 0, 10) == pytest.approx(0.4)
    assert np.isnan(M.band_share(_img(), track, 0, 10))


def test_cross_ratio_and_cross_negative_by_hand():
    a = _img()
    f1, f2 = 100 / C.NFFT, 140 / C.NFFT
    a[:, 120] = 2.0                                                            # Mitte
    a[:, 100] = 4.0                                                            # Ton
    assert M.cross_ratio(a, f1, f2, 0, 10) == pytest.approx(0.5)
    a[3, 122] = -6.0                                                           # negativer Wert im Band um die Mitte
    assert M.cross_negative(a, f1, f2, 0, 10) == pytest.approx(-6.0 / 4.0)
    assert M.cross_negative(a, f1, f2, 0, 3) == pytest.approx(0.0)
    b = _img()
    b[:, 100], b[:, 123] = 5.0, 5.0                                            # 123 liegt außerhalb von 120 ± 3 + 0: Band 117..123 enthält es
    assert M.cross_ratio(b, f1, f2, 0, 10) == pytest.approx(1.0)
    b[:, 123] = 0.0
    b[:, 124] = 5.0
    assert M.cross_ratio(b, f1, f2, 0, 10) == 0.0
    assert np.isnan(M.cross_ratio(_img(), f1, f2, 0, 10)) and np.isnan(M.cross_negative(_img(), f1, f2, 0, 10))


def test_negative_depth():
    a = _img()
    a[:, 5] = 4.0
    assert M.negative_depth(a, 0, 10) == 0.0
    a[2, 30] = -2.0
    assert M.negative_depth(a, 0, 10) == pytest.approx(-0.5) and M.negative_depth(a, 3, 10) == 0.0 and M.negative_depth(_img(), 0, 10) == 0.0


def test_rmse_by_hand():
    est = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    truth = np.array([0.1, 0.1, 0.3, 0.5, 0.9])
    assert M.rmse(est, truth, 0, 5) == pytest.approx(np.sqrt((0.01 + 0.01 + 0.16) / 5)) and M.rmse(est, truth, 1, 4) == pytest.approx(np.sqrt(0.02 / 3)) and M.rmse(est, est, 0, 5) == 0.0


def test_peak_estimates_of_a_tone_and_the_error_bookkeeping():
    n = 512
    sig = SG.tone(n, 120 / 1024)
    z = SG.analytic(sig.x)
    est = M.peak_estimates(z, 12.0, abs_floor=0.5)
    assert set(est) == {"Spektrogramm", "Neu zugeordnet", "Neu zugeordnet, Zeit", "Synchrosqueezing", "Synchrosqueezing mit Schwelle", "Wigner-Ville", "SPWVD"}
    assert (est["Spektrogramm"][100:400] == 120 / 1024).all() and est["Neu zugeordnet"][100:400] == pytest.approx(120 / 1024, abs=1e-6)
    assert est["Wigner-Ville"][256] == 120 / 1024 and est["Synchrosqueezing"][100:400] == pytest.approx(120 / 1024)
    err = M.ridge_errors(est, sig.tracks[0], n)
    assert err["Spektrogramm"] == 0.0 and err["Neu zugeordnet"] < 1e-6 and "Neu zugeordnet, Zeit" not in err and err["Neu zugeordnet (zur neuen Zeit)"] < 1e-6
    assert "Wigner-Ville" not in M.peak_estimates(z, 12.0, with_wvd=False) and "Synchrosqueezing mit Schwelle" not in M.peak_estimates(z, 12.0)


def test_ridge_error_against_the_frame_time_and_the_reassigned_time_for_a_chirp():
    n = 512
    sig = SG.linear_chirp(n, 0.05, 0.20)
    err = M.ridge_errors(M.peak_estimates(SG.analytic(sig.x), 12.0, with_wvd=False), sig.tracks[0], n)
    assert err["Neu zugeordnet (zur neuen Zeit)"] < 1e-4 < 2e-4 < err["Spektrogramm"] and err["Neu zugeordnet"] < 5e-4


def test_method_image_shapes_ridges_and_validation():
    n = 256
    sig = SG.linear_chirp(n, 0.05, 0.2)
    z = SG.analytic(sig.x)
    for method in C.METHODS:
        img, ridge = M.method_image(z, method, 8.0)
        assert img.shape == (n, H) and ridge.shape == (n,) and np.abs(ridge[80:180] - sig.tracks[0][80:180]).max() < 0.004
    assert (M.method_image(z, "Spektrogramm", 8.0)[0] >= 0).all() and (M.method_image(z, "Synchrosqueezing", 8.0)[0] >= 0).all()
    assert M.method_image(z, "Wigner-Ville", 8.0)[0].min() < 0
    with pytest.raises(ValueError):
        M.method_image(z, "Hough", 8.0)


def test_gate_and_floor_change_the_reassigned_image():
    rng = G.SplitMix64(31)
    z = SG.analytic(SG.linear_chirp(256, 0.05, 0.2).x + 0.5 * rng.normal(256))
    loose, _ = M.method_image(z, "Neu zugeordnet", 8.0, floor_db=-80.0)
    tight, _ = M.method_image(z, "Neu zugeordnet", 8.0, floor_db=-20.0)
    gated, _ = M.method_image(z, "Neu zugeordnet", 8.0, floor_db=-80.0, abs_floor=3.0)
    assert tight.sum() < loose.sum() and gated.sum() < loose.sum()


def test_band_share_with_tiny_magnitudes_and_empty_images():
    a = _img()
    a[:, 100] = 0.1
    a[:, 200] = 0.1                                                            # Gesamtbetrag 2 * 10 * 0,1 = 2; auf der Linie die Hälfte
    assert M.band_share(a, [np.full(10, 100 / C.NFFT)], 0, 10) == pytest.approx(0.5)
    b = _img()
    b[:, 100] = 1e-3
    assert M.band_share(b, [np.full(10, 100 / C.NFFT)], 0, 10) == pytest.approx(1.0)          # Gesamtbetrag 0,01 liegt unter 1


def test_count_peaks_in_a_profile_of_exactly_three_points():
    assert M.count_peaks([0, 1, 0]) == 1 and M.count_peaks([0, 0, 0]) == 0 and M.count_peaks([2, 1, 2]) == 0


def test_cross_ratio_and_negative_with_small_values_and_a_zero_tone_band():
    f1, f2 = 100 / C.NFFT, 140 / C.NFFT
    a = _img()
    a[:, 120] = 0.1
    a[:, 100] = 0.4                                                            # Nenner 0,4 liegt unter 1
    assert M.cross_ratio(a, f1, f2, 0, 10) == pytest.approx(0.25)
    a[2, 121] = -0.2
    assert M.cross_negative(a, f1, f2, 0, 10) == pytest.approx(-0.5)
    b = _img()
    b[:, 120] = 3.0
    b[4, 118] = -1.0                                                           # im Band um den Ton steht nichts: undefiniert (nan), nicht unendlich
    assert np.isnan(M.cross_ratio(b, f1, f2, 0, 10)) and np.isnan(M.cross_negative(b, f1, f2, 0, 10))


def test_negative_depth_with_small_values():
    a = _img()
    a[:, 5] = 0.5
    a[3, 9] = -0.25
    assert M.negative_depth(a, 0, 10) == pytest.approx(-0.5)


def test_ridge_errors_time_window_of_the_reassigned_time_is_inclusive_at_both_ends():
    n = 16                                                                     # Zeitfenster der Fehler: 2 bis 13
    truth = np.arange(n) * 0.01
    tt = np.full(n, 5.0)
    tt[2], tt[3], tt[4], tt[5] = 0.0, 15.0, -0.5, 15.5                         # 0 und n - 1 zählen, -0,5 und 15,5 nicht
    delta = np.full(n, 0.001)
    delta[2], delta[3], delta[4], delta[5] = 0.003, 0.002, 1.0, 1.0
    est = {"Spektrogramm": truth.copy(), "Neu zugeordnet": np.interp(tt, np.arange(n), truth) + delta, "Neu zugeordnet, Zeit": tt}
    err = M.ridge_errors(est, truth, n)
    assert err["Neu zugeordnet (zur neuen Zeit)"] == pytest.approx(np.sqrt((0.003 ** 2 + 0.002 ** 2 + 8 * 0.001 ** 2) / 10)) and err["Spektrogramm"] == 0.0


def test_time_windows_scale_with_the_length_for_other_lengths():
    assert M.inner(4096) == (800, 3296) and M.ridge_window(4096) == (480, 3616) and M.inner(1024) == (200, 824) and M.ridge_window(256) == (30, 226)
