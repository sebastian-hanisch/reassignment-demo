"""Neuzuordnung und Synchrosqueezing gegen bekannte Schwerpunkte (Impuls, Ton, linearer Chirp) und gegen Energie- und Summenbilanzen."""

import math

import numpy as np
import pytest

import rea_constants as C
import rea_reassign as R
import rea_signals as SG
import rea_stft as T
import signal_generator as G


def test_an_impulse_is_reassigned_to_its_own_time():
    z = np.zeros(128, complex)
    z[60] = 1.0
    X, f_hat, t_hat, mask = R.reassign_fields(z, 6.0, 256, -40.0)
    assert mask.any() and np.unique(np.round(t_hat[mask], 9)).tolist() == [60.0]


def test_a_tone_is_reassigned_to_its_frequency():
    n, f0 = 256, 0.137
    z = np.exp(2j * np.pi * f0 * np.arange(n))
    X, f_hat, t_hat, mask = R.reassign_fields(z, 12.0, 1024, -30.0)
    inner = (slice(64, 192), slice(None))
    fh = f_hat[inner][mask[inner]]
    assert abs(fh.mean() - f0) < 1e-6 and fh.std() < 5e-5 and mask[inner].any()


def test_a_linear_chirp_is_reassigned_onto_its_line():
    """Das Gauß-Fenster macht den Schwerpunkt für den linearen Chirp exakt: f_hat = f(t_hat) bis auf Randeffekte und Rasterung."""
    n = 512
    sig = SG.linear_chirp(n, 0.05, 0.20)
    z = SG.analytic(sig.x)
    X, f_hat, t_hat, mask = R.reassign_fields(z, 12.0, 1024, -30.0)
    mask[:80] = False
    mask[-80:] = False
    err = (f_hat - (0.05 + (0.20 - 0.05) / (n - 1) * t_hat))[mask]
    assert mask.sum() > 1000 and np.abs(err).max() < 5e-4 and err.std() < 5e-5
    frame_err = (f_hat - sig.tracks[0][:, None])[mask]
    assert frame_err.std() > 10 * err.std()          # gegen die Rahmenzeit scheint der Fehler groß: die Punkte wandern entlang der Linie


def test_the_reassigned_image_keeps_the_energy_of_the_points_inside_the_grid():
    sig = SG.linear_chirp(256, 0.05, 0.2)
    z = SG.analytic(sig.x)
    X, f_hat, t_hat, mask = R.reassign_fields(z, 8.0, 512, -50.0)
    img = R.reassigned_from(X, t_hat, f_hat, mask, 512)
    ti, fi = np.rint(t_hat).astype(int), np.rint(f_hat * 512).astype(int)
    ok = mask & (ti >= 0) & (ti < 256) & (fi >= 0) & (fi < 512)
    assert img.shape == (256, 512) and img.sum() == pytest.approx((np.abs(X) ** 2)[ok].sum(), rel=1e-12) and (img >= 0).all()
    assert R.reassigned_spectrogram(z, 8.0, 512, -50.0) == pytest.approx(img)


def test_reassignment_commutes_with_a_shift_of_the_input():
    """Ein Chirp-Burst mit Gauß-Hüllkurve, weit von den Rändern: Verschieben der Eingabe um 21 verschiebt das Bild um 21 (Zeilen)."""
    t = np.arange(300)
    x = np.exp(-((t - 130) ** 2) / (2 * 30.0 ** 2)) * np.cos(2 * np.pi * (0.08 * t + 0.0002 * t * t))
    z = SG.analytic(x)
    z[:20] = 0.0
    z[-60:] = 0.0
    a = R.reassigned_spectrogram(z, 8.0, 256, -50.0)
    b = R.reassigned_spectrogram(np.roll(z, 21), 8.0, 256, -50.0)
    assert b[21:279] == pytest.approx(a[0:258], rel=1e-9, abs=1e-12 * a.max())


def test_synchrosqueezing_keeps_the_sum_over_frequency_of_the_stft():
    sig = SG.linear_chirp(256, 0.05, 0.2)
    z = SG.analytic(sig.x)
    X, f_hat, _t, _mask = R.reassign_fields(z, 8.0, 512, -300.0)
    full = np.ones(X.shape, bool)
    fi = np.rint(f_hat * 512).astype(int)
    inside = (fi >= 0) & (fi < 512)
    S = R.squeezed_from(X, f_hat, full, 512)
    assert S.shape == (256, 512) and np.iscomplexobj(S)
    assert S.sum(axis=1) == pytest.approx((X * inside).sum(axis=1), rel=1e-9, abs=1e-9)
    assert R.synchrosqueeze(z, 8.0, 512, -300.0) == pytest.approx(S)


def test_synchrosqueezing_of_a_tone_is_a_single_line():
    n = 256
    z = np.exp(2j * np.pi * 0.125 * np.arange(n))
    S = R.synchrosqueeze(z, 8.0, 512, -40.0)
    inner = np.abs(S[64:192])
    assert (np.argmax(inner, axis=1) == 64).all() and inner[:, 64].min() > 50 * np.delete(inner, 64, axis=1).max()


def test_the_power_floor_and_the_absolute_floor_drop_points():
    z = SG.analytic(SG.tone(256, 0.1).x)
    _X, _f, _t, loose = R.reassign_fields(z, 8.0, 512, -80.0)
    _X, _f, _t, tight = R.reassign_fields(z, 8.0, 512, -20.0)
    X, _f, _t, gated = R.reassign_fields(z, 8.0, 512, -80.0, abs_floor=0.5 * np.abs(X_max := T.stft3(z, 8.0, 512)[0]).max())
    assert tight.sum() < loose.sum() and gated.sum() < loose.sum() and (tight <= loose).all() and (gated <= loose).all()
    assert (np.abs(X)[gated] > 0.5 * np.abs(X_max).max()).all()


def test_noise_floor_formula_and_its_meaning_by_simulation():
    _k, g, _dg, _tg = T.gauss_windows(12.0)
    sd = 0.3
    assert R.noise_floor(12.0, sd) == pytest.approx(3.0 * 2.0 * sd * math.sqrt(np.sum(g ** 2)), rel=1e-14)
    assert R.noise_floor(12.0, sd, factor=1.0) * 3.0 == pytest.approx(R.noise_floor(12.0, sd), rel=1e-14)
    # E|X|^2 = 4 sd^2 sum g^2 für die STFT von analytischem weißem Rauschen (Mittel über viele Zeitpunkte und Läufe)
    rng = G.SplitMix64(11)
    power = []
    for _ in range(6):
        z = SG.analytic(sd * rng.normal(1024))
        power.append((np.abs(T.stft3(z, 12.0, 1024)[0][100:-100, 20:480]) ** 2).mean())
    assert np.mean(power) == pytest.approx(4.0 * sd ** 2 * np.sum(g ** 2), rel=0.08)


def test_a_zero_input_has_finite_fields_and_an_empty_mask():
    """Ohne Energie ist der Schwerpunkt unbestimmt: f_hat = f, t_hat = t (Quotient durch 1 ersetzt), nichts wird neu zugeordnet."""
    z = np.zeros(64, complex)
    X, f_hat, t_hat, mask = R.reassign_fields(z, 4.0, 128, -60.0)
    assert np.isfinite(f_hat).all() and np.isfinite(t_hat).all() and not mask.any()
    assert f_hat == pytest.approx(np.broadcast_to(np.arange(128) / 128, (64, 128))) and t_hat == pytest.approx(np.broadcast_to(np.arange(64)[:, None], (64, 128)))
    assert R.reassigned_spectrogram(z, 4.0, 128).sum() == 0 and np.abs(R.synchrosqueeze(z, 4.0, 128)).sum() == 0


def test_the_power_floor_is_strictly_below_the_peak():
    """Maske: Leistung über floor_db gegen die Spitze, also echt darüber; bei 0 dB bleibt nichts (die Spitze selbst liegt nicht darüber)."""
    z = SG.analytic(SG.tone(128, 0.125).x)
    assert not R.reassign_fields(z, 6.0, 256, 0.0)[3].any() and R.reassign_fields(z, 6.0, 256, -1.0)[3].any()
