"""Auswertung: Einstellungen, Messung, Analyse (feste Werte der Presets) und die Experimentfunktionen (kleine Instanzen)."""

import math
import pickle

import numpy as np
import pytest

import rea_constants as C
import rea_evaluation as ev
import rea_reassign as R
import rea_signals as SG
import signal_generator as G


def _run(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], sigma=p["sigma"], method=p["method"], df=p["df"], noise=p["noise"], snr=p["snr"], sigma_t=p["sigma_t"], sigma_lag=p["sigma_lag"], floor=p["floor"], gate=p["gate"],
                seed=p["seed"])
    return ev.analyse(ev.Settings(**{**base, **over}))


def _vals(a):
    f = lambda v: -1.0 if v is None else v          # noqa: E731
    return (a.width, a.width_spec, f(a.rmse), f(a.rmse_spec), a.peaks, a.peaks_spec, a.share, a.share_spec, f(a.cross), f(a.cross_spec), a.neg, a.gate_value)


P = list(C.PRESETS)


def test_settings_flags_and_the_decimal_comma():
    s = ev.Settings()
    assert (s.n, s.signal, s.sigma, s.method, s.noise, s.gate, s.seed) == (512, "Chirp", 12, "Neu zugeordnet", "aus", "aus", C.DEFAULT_SEED)
    assert s.noise_on is False and s.two is False and ev.Settings(signal="Zwei Töne").two is True and ev.Settings(signal="Zwei parallele Chirps").two is True
    # die Rauschschwelle wirkt nur mit Rauschen und nur bei Neuzuordnung und Synchrosqueezing
    assert ev.Settings(noise="an", gate="an").gate_on is True and ev.Settings(noise="an", gate="an", method="Synchrosqueezing").gate_on is True
    assert ev.Settings(noise="aus", gate="an").gate_on is False and ev.Settings(noise="an", gate="aus").gate_on is False and ev.Settings(noise="an", gate="an", method="Wigner-Ville").gate_on is False
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0" and ev.de(float("nan")) == "–"
    assert ev.window_width(12.0) == pytest.approx(1 / (2 * math.pi * 12)) and ev.window_width(12.0) == pytest.approx(0.0132629, abs=1e-7)


def test_measurement_is_the_signal_plus_scaled_noise():
    sig, x, z, sd = ev.build_measurement(ev.Settings(n=256, signal="Sinus-FM", noise="aus"))
    assert sd == 0.0 and x.tolist() == sig.x.tolist() and z == pytest.approx(SG.analytic(sig.x))
    s = ev.Settings(n=256, signal="Chirp", noise="an", snr=6.0, seed=5)
    sig, x, z, sd = ev.build_measurement(s)
    assert sd == pytest.approx(SG.noise_sigma(6.0)) and x - sig.x == pytest.approx(sd * G.SplitMix64(5).normal(256)) and z == pytest.approx(SG.analytic(x))
    assert ev.build_measurement(ev.Settings(n=256, noise="an", seed=6))[1].tolist() != ev.build_measurement(ev.Settings(n=256, noise="an", seed=5))[1].tolist()


def test_analyse_the_five_presets_golden():
    assert _vals(_run(P[0])) == pytest.approx((1.114539, 24.915486, 1.84e-05, 0.0002821, 32, 1, 1.0, 0.275359, -1.0, -1.0, 0.0, 0.0), abs=1e-6)
    assert _vals(_run(P[1])) == pytest.approx((11.418482, 40.737791, 0.0053012, 0.0045401, 2, 3, 0.180124, 0.168784, -1.0, -1.0, 0.0, 0.0), abs=1e-6)
    assert _vals(_run(P[2])) == pytest.approx((11.24623, 48.176101, -1.0, -1.0, 3, 2, 0.327402, 0.284386, 1.2979544, 0.0004915, -0.998764, 0.0), abs=1e-6)
    assert _vals(_run(P[3])) == pytest.approx((6.460728, 38.181177, -1.0, -1.0, 4, 2, 0.464608, 0.314986, 0.3670044, 0.970274, 0.0, 0.0), abs=1e-6)
    assert _vals(_run(P[4])) == pytest.approx((3.394666, 50.72262, 0.0802086, 0.001531, 4, 7, 0.669141, 0.137781, -1.0, -1.0, 0.0, 0.0), abs=1e-6)


def test_analyse_variants_golden():
    assert _vals(_run(P[4], gate="an")) == pytest.approx((1.916617, 50.72262, 0.0014909, 0.001531, 3, 7, 0.947396, 0.137781, -1.0, -1.0, 0.0, 19.566554), abs=1e-6)
    assert _vals(_run(P[4], method="Neu zugeordnet", gate="an")) == pytest.approx((2.135175, 50.72262, 0.0015132, 0.001531, 8, 7, 0.909988, 0.137781, -1.0, -1.0, 0.0, 19.566554), abs=1e-6)
    assert _vals(_run(P[0], signal="Zwei parallele Chirps", df=0.05)) == pytest.approx((2.259434, 48.391328, -1.0, -1.0, 31, 2, 0.96581, 0.283009, -1.0, -1.0, 0.0, 0.0), abs=1e-6)
    assert _vals(_run(P[0], signal="Zwei parallele Chirps", df=0.05, method="SPWVD")) == pytest.approx((23.120333, 48.391328, -1.0, -1.0, 2, 2, 0.541136, 0.283009, -1.0, -1.0, -0.357187, 0.0), abs=1e-6)
    assert _vals(_run(P[0], signal="Ton", method="Spektrogramm")) == pytest.approx((24.085474, 24.085474, 0.0002813, 0.0002813, 1, 1, 0.284458, 0.284458, -1.0, -1.0, 0.0, 0.0), abs=1e-6)
    assert _vals(_run(P[0], signal="Sinus-FM", method="SPWVD", sigma_t=0)) == pytest.approx((25.73152, 40.737791, 0.0059758, 0.0045401, 9, 3, 0.21824, 0.168784, -1.0, -1.0, -0.377479, 0.0), abs=1e-6)


def test_the_gate_value_is_the_noise_floor_and_only_set_when_active():
    s = ev.Settings(signal="Chirp", method="Synchrosqueezing", noise="an", snr=0.0, gate="an")
    assert ev.analyse(s).gate_value == pytest.approx(R.noise_floor(12.0, SG.noise_sigma(0.0)), rel=1e-12)
    assert ev.analyse(ev.Settings(signal="Chirp", method="Synchrosqueezing", noise="an", snr=0.0, gate="aus")).gate_value == 0.0
    assert ev.analyse(ev.Settings(signal="Chirp", method="Synchrosqueezing", noise="aus", gate="an")).gate_value == 0.0


def test_structure_of_the_result_and_the_single_versus_two_component_fields():
    a = _run(P[0])
    assert a.img.shape == a.img_spec.shape == (512, 512) and a.ridge.shape == a.ridge_spec.shape == a.x.shape == (512,) and (a.lo, a.hi) == (100, 412) and a.cross is None and a.rmse is not None
    b = _run(P[2])
    assert b.rmse is None and b.rmse_spec is None and b.cross is not None and b.signal.name == "Zwei Töne"
    c = _run(P[0], signal="Zwei parallele Chirps", df=0.05)
    assert c.rmse is None and c.cross is None and len(c.signal.tracks) == 2
    assert _run(P[0], n=256).img.shape == (256, 512) and (_run(P[0], n=256).lo, _run(P[0], n=256).hi) == (50, 206)


def test_the_spectrogram_side_does_not_depend_on_the_method():
    ref = _run(P[0], method="Spektrogramm")
    for m in C.METHODS:
        a = _run(P[0], method=m)
        assert a.img_spec == pytest.approx(ref.img) and a.width_spec == pytest.approx(ref.width) and a.ridge_spec.tolist() == ref.ridge.tolist()


def test_analysis_is_picklable():
    a = ev.analyse(ev.Settings(n=256, signal="Zwei Töne"))
    b = pickle.loads(pickle.dumps(a))
    assert b.width == a.width and b.settings == a.settings and (b.img == a.img).all()


def test_sweep_concentration_small_instance():
    rows = ev.sweep_concentration(n=256, sigmas=(6.0, 12.0))
    assert [r["signal"] for r in rows] == ["Chirp", "Sinus-FM", "Ton"]
    got = [(round(r["spec"][6.0], 4), round(r["spec"][12.0], 4), round(r["reassigned"], 4), round(r["squeezed"], 4), round(r["wvd"], 4), round(r["spwvd"], 4)) for r in rows]
    assert got == pytest.approx([(48.5756, 27.2784, 1.2272, 5.1543, 9.9446, 13.6786), (51.6538, 40.9405, 3.5621, 11.4677, 30.6479, 31.3246), (48.1538, 24.0855, 1.0, 1.0, 9.7326, 10.2582)], abs=1e-4)


def test_sweep_cross_small_instance():
    r = ev.sweep_cross(n=256, smoothings=((0.0, 40.0), (6.0, 20.0)))
    assert (r["f1"], r["f2"]) == (0.1, 0.18) and r["period"] == pytest.approx(12.5)
    got = [(m["method"], m["ratio"], round(m["width"], 4)) for m in r["methods"]]
    assert [g[0] for g in got] == ["Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Wigner-Ville"]
    assert [g[1] for g in got] == pytest.approx([0.00049187, 4.5047e-06, 9.1046e-08, 1.2827858], rel=1e-4) and [g[2] for g in got] == pytest.approx([48.1761, 2.0010, 1.9717, 18.1941], abs=1e-4)
    assert [(x["sigma_t"], x["sigma_lag"], round(x["ratio"], 4), round(x["width"], 3)) for x in r["smooth"]] == pytest.approx([(0.0, 40.0, 1.2759, 14.821), (6.0, 20.0, 0.0135, 20.669)], abs=1e-3)
    assert r["negative"] == pytest.approx(-1.95021, abs=1e-4)


def test_sweep_noise_small_instance():
    rows = ev.sweep_noise(n=256, snrs=(20, 0), runs=2)
    assert [(r["snr"], r["runs"]) for r in rows] == [(20, 2), (0, 2)]
    assert [round(rows[0][m], 6) for m in ev.NOISE_METHODS] == pytest.approx([0.000339, 0.000189, 0.001273, 0.001274, 0.000284, 0.000345], abs=1e-6)
    assert [round(rows[1][m], 6) for m in ev.NOISE_METHODS] == pytest.approx([0.001156, 0.001122, 0.075137, 0.001512, 0.000437, 0.000591], abs=1e-6)


def test_sweep_resolution_small_instance():
    rows = ev.sweep_resolution(n=256, dfs=(0.02, 0.05))
    assert [(r["peaks_spec"], r["peaks_re"]) for r in rows] == [(2, 6), (2, 2)]
    assert [round(r["widths"], 4) for r in rows] == [1.508, 3.7699] and [round(r["share_spec"], 4) for r in rows] == [0.3187, 0.2844] and [round(r["share_re"], 4) for r in rows] == [0.4903, 0.9667]


def test_sweep_window_small_instance():
    rows = ev.sweep_window(n=256, sigmas=(8, 16), runs=2)
    assert [(r["sigma"], r["runs"], r["snr"]) for r in rows] == [(8, 2, 10), (16, 2, 10)]
    keys = ("Spektrogramm", "Neu zugeordnet", "Neu zugeordnet (zur neuen Zeit)", "Synchrosqueezing")
    assert [round(rows[0]["clean"][k], 6) for k in keys] == pytest.approx([0.002537, 0.002507, 0.002917, 0.002866], abs=1e-6)
    assert [round(rows[0]["noisy"][k], 6) for k in keys] == pytest.approx([0.002334, 0.002339, 0.002704, 0.056467], abs=1e-6)
    assert [round(rows[1]["clean"][k], 6) for k in keys] == pytest.approx([0.008291, 0.008301, 0.007654, 0.013426], abs=1e-6)
    assert [round(rows[1]["noisy"][k], 6) for k in keys] == pytest.approx([0.009131, 0.00914, 0.007581, 0.038213], abs=1e-6)
