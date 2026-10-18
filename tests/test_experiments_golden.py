"""Die Experimente in voller Größe gegen feste Werte (feste Ströme); die Tabellen der App und der README stehen auf denselben Zahlen."""

import pytest

import rea_evaluation as ev
from sweeps_cache import get


def test_concentration_golden():
    rows = {r["signal"]: r for r in get("concentration")}
    got = {k: (r["spec"][6.0], r["spec"][12.0], r["spec"][24.0], r["reassigned"], r["squeezed"], r["wvd"], r["spwvd"]) for k, r in rows.items()}
    assert got["Chirp"] == pytest.approx((48.26, 24.92, 17.57, 1.11, 1.88, 5.71, 11.19), abs=0.01)
    assert got["Sinus-FM"] == pytest.approx((51.63, 40.74, 39.68, 3.56, 11.42, 21.83, 31.23), abs=0.01)
    assert got["Ton"] == pytest.approx((48.15, 24.09, 12.06, 1.00, 1.00, 5.59, 10.24), abs=0.01)


def test_cross_golden():
    r = get("cross")
    assert [m["method"] for m in r["methods"]] == ["Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Wigner-Ville"]
    assert [m["ratio"] for m in r["methods"]] == pytest.approx([4.9145e-4, 4.327e-6, 9.1e-8, 1.297954], rel=1e-3)
    assert [m["width"] for m in r["methods"]] == pytest.approx([48.18, 2.00, 1.97, 11.25], abs=0.01)
    assert [(x["sigma_t"], x["sigma_lag"], round(x["ratio"], 4), round(x["width"], 2)) for x in r["smooth"]] == pytest.approx(
        [(0.0, 255.0, 1.2959, 10.29), (0.0, 40.0, 1.2762, 12.23), (0.0, 20.0, 1.2729, 24.06), (6.0, 20.0, 0.0134, 20.63), (12.0, 20.0, 0.0, 20.49), (12.0, 10.0, 0.0, 40.88)], abs=0.01)
    assert r["negative"] == pytest.approx(-2.0163, abs=1e-3) and r["period"] == pytest.approx(12.5) and (r["f1"], r["f2"]) == (0.10, 0.18)


def test_noise_golden():
    rows = get("noise")
    assert [r["snr"] for r in rows] == [60, 30, 20, 10, 0, -5] and all(r["runs"] == 20 for r in rows)
    want = {60: (0.00028, 0.00002, 0.00028, 0.00028, 0.00028, 0.00028), 30: (0.00029, 0.00005, 0.00030, 0.00030, 0.00028, 0.00028), 20: (0.00032, 0.00016, 0.00033, 0.00033, 0.00028, 0.00029),
            10: (0.00059, 0.00051, 0.00060, 0.00060, 0.00029, 0.00038), 0: (0.00167, 0.00164, 0.05399, 0.00683, 0.00101, 0.00083), -5: (0.06748, 0.06748, 0.13676, 0.10609, 0.05948, 0.02564)}
    for r in rows:
        assert [r[m] for m in ev.NOISE_METHODS] == pytest.approx(want[r["snr"]], abs=6e-6), r["snr"]


def test_resolution_golden():
    rows = get("resolution")
    assert [r["df"] for r in rows] == [0.01, 0.015, 0.02, 0.03, 0.05]
    assert [(r["peaks_spec"], r["peaks_re"]) for r in rows] == [(1, 1), (1, 2), (2, 6), (2, 2), (2, 2)]
    assert [round(r["share_spec"], 4) for r in rows] == pytest.approx([0.4530, 0.3722, 0.3186, 0.2862, 0.2846], abs=1e-4) and [round(r["share_re"], 4) for r in rows] == pytest.approx([0.4457, 0.4007, 0.4965, 0.7168, 0.9671], abs=1e-4)
    assert [round(r["widths"], 2) for r in rows] == [0.75, 1.13, 1.51, 2.26, 3.77]


def test_window_golden():
    rows = get("window")
    keys = ("Spektrogramm", "Neu zugeordnet", "Neu zugeordnet (zur neuen Zeit)", "Synchrosqueezing")
    clean = {4: (0.00090, 0.00086, 0.00087, 0.00094), 6: (0.00172, 0.00174, 0.00185, 0.00210), 8: (0.00261, 0.00258, 0.00298, 0.00293), 12: (0.00454, 0.00453, 0.00535, 0.00530),
             16: (0.00822, 0.00823, 0.00775, 0.01413), 24: (0.01279, 0.01277, 0.01041, 0.02705), 32: (0.01322, 0.01319, 0.00854, 0.03202)}
    noisy = {4: (0.00266, 0.00264, 0.00260, 0.00854), 6: (0.00233, 0.00231, 0.00232, 0.01214), 8: (0.00291, 0.00289, 0.00313, 0.03295), 12: (0.00510, 0.00509, 0.00538, 0.05638),
             16: (0.00911, 0.00911, 0.00768, 0.05196), 24: (0.01251, 0.01251, 0.01001, 0.02733), 32: (0.01415, 0.01414, 0.00921, 0.03065)}
    assert [r["sigma"] for r in rows] == list(clean) and all(r["runs"] == 20 and r["snr"] == 10 for r in rows)
    for r in rows:
        assert [r["clean"][k] for k in keys] == pytest.approx(clean[r["sigma"]], abs=6e-6) and [r["noisy"][k] for k in keys] == pytest.approx(noisy[r["sigma"]], abs=6e-6)
