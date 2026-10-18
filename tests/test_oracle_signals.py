"""Testsignale gegen ihre Definition; wahre Momentanfrequenz gegen die numerische Ableitung der Phase; Rauschpegel gegen die Definition."""

import math

import numpy as np
import pytest

import rea_constants as C
import rea_signals as SG
import signal_generator as G


def _phase_derivative(x_analytic):
    """Momentanfrequenz aus der entfalteten Phase des analytischen Signals (zentrale Differenz), Zyklen je Abtastwert."""
    ph = np.unwrap(np.angle(x_analytic))
    return (ph[2:] - ph[:-2]) / (2.0 * 2.0 * np.pi)


@pytest.mark.parametrize("kind", ["Ton", "Chirp", "Sinus-FM"])
def test_true_frequency_is_the_derivative_of_the_phase(kind):
    sig = SG.make_signal(kind, 512)
    est = _phase_derivative(SG.analytic(sig.x))
    assert len(sig.tracks) == 1 and sig.tracks[0].shape == (512,)
    # Ton und Chirp: nahezu exakt; Sinus-FM: das analytische Signal des reellen Cosinus weicht bei starker Modulation um bis zu 0,003 von der Frequenz der komplexen Komponente ab (Bedrosian)
    assert np.abs(est[30:-30] - sig.tracks[0][1:-1][30:-30]).max() < (4e-3 if kind == "Sinus-FM" else 2e-3)


def test_signals_are_unit_amplitude_and_bounded_in_frequency():
    for kind in C.SIGNALS:
        sig = SG.make_signal(kind, 256, 0.12)
        assert sig.x.shape == (256,) and np.isrealobj(sig.x)
        assert all(0.0 < tr.min() and tr.max() < 0.25 for tr in sig.tracks)
        assert len(sig.tracks) == (2 if kind in C.SIGNAL_TWO else 1) and sig.name == kind
    assert np.abs(SG.make_signal("Chirp", 512).x).max() <= 1.0 and np.abs(SG.make_signal("Zwei Töne", 512).x).max() <= 2.0


def test_chirp_goes_linearly_from_f0_to_f1():
    sig = SG.linear_chirp(101, 0.05, 0.20)
    tr = sig.tracks[0]
    assert tr[0] == 0.05 and tr[-1] == pytest.approx(0.20) and tr[50] == pytest.approx(0.125) and np.diff(tr) == pytest.approx(0.0015)


def test_sine_fm_frequency_formula():
    sig = SG.sine_fm(256, 0.12, 0.04, 0.01)
    t = np.arange(256)
    assert sig.tracks[0] == pytest.approx(0.12 + 0.04 * np.sin(2 * np.pi * 0.01 * t)) and sig.tracks[0].max() <= 0.16 + 1e-12 and sig.tracks[0].min() >= 0.08 - 1e-12


def test_two_tones_and_parallel_chirps_have_the_requested_distance():
    two = SG.make_signal("Zwei Töne", 256, 0.05)
    assert two.tracks[1] - two.tracks[0] == pytest.approx(0.05) and two.tracks[0][0] == C.TWO_F1
    par = SG.make_signal("Zwei parallele Chirps", 256, 0.07)
    assert par.tracks[1] - par.tracks[0] == pytest.approx(0.07) and par.tracks[0][0] == C.PAR_F0 and par.tracks[0][-1] == pytest.approx(C.PAR_F1)
    assert two.x == pytest.approx(np.cos(2 * np.pi * 0.10 * np.arange(256) + 0.3) + np.cos(2 * np.pi * 0.15 * np.arange(256) + 0.6))


def test_unknown_signal_is_rejected():
    with pytest.raises(ValueError):
        SG.make_signal("Rechteck", 128)


def test_noise_sigma_by_definition_and_by_simulation():
    assert SG.noise_sigma(0.0) == pytest.approx(math.sqrt(0.5)) and SG.noise_sigma(10.0) == pytest.approx(math.sqrt(0.05)) and SG.noise_sigma(-10.0) == pytest.approx(math.sqrt(5.0))
    # SNR gegen die Linie der Leistung 1/2: gemessenes Verhältnis aus Signal- und Rauschleistung
    rng = G.SplitMix64(21)
    sig = SG.linear_chirp(4096, 0.05, 0.2)
    noise = SG.noise_sigma(6.0) * rng.normal(4096)
    assert 10 * math.log10(np.mean(sig.x ** 2) / np.mean(noise ** 2)) == pytest.approx(6.0, abs=0.4)
