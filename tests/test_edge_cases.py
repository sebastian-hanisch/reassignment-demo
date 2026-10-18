"""Randfälle: alle Signale, Längen, Fensterbreiten, Verfahren, Rauschabstände, Abbildungen und PDF."""

import numpy as np
import pytest

import rea_constants as C
import rea_evaluation as ev
import rea_visualization as V
from rea_pdf_export import build_pdf


def _finite(a):
    return np.isfinite(a.img).all() and np.isfinite(a.img_spec).all() and np.isfinite(a.ridge).all() and np.isfinite(a.width) and 0.0 <= a.share <= 1.0 and (a.rmse is None or a.rmse >= 0)


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_every_signal_runs_for_the_shortest_and_the_longest_record_and_every_method(signal):
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        for method in C.METHODS:
            a = ev.analyse(ev.Settings(n=n, signal=signal, method=method))
            assert _finite(a) and a.img.shape == (n, 512) and (a.cross is None) == (signal != "Zwei Töne") and (a.rmse is None) == (signal in C.SIGNAL_TWO)


@pytest.mark.parametrize("method", list(C.METHODS))
def test_extreme_windows_distances_and_noise_levels_run(method):
    for sigma in (C.SIGMA_MIN, C.SIGMA_MAX):
        for df in (C.DF_MIN, C.DF_MAX):
            for snr in (C.SNR_MIN, C.SNR_MAX):
                for gate in C.GATE_MODES:
                    a = ev.analyse(ev.Settings(n=256, signal="Zwei Töne", sigma=sigma, method=method, df=df, noise="an", snr=snr, gate=gate, sigma_t=C.SIGMA_T_MAX, sigma_lag=C.SIGMA_LAG_MIN, floor=C.FLOOR_MAX))
                    assert _finite(a) and a.gate_value >= 0.0


def test_parallel_chirps_at_the_largest_distance_stay_below_the_aliasing_limit():
    a = ev.analyse(ev.Settings(signal="Zwei parallele Chirps", df=C.DF_MAX, method="Wigner-Ville"))
    assert max(t.max() for t in a.signal.tracks) < 0.25 and _finite(a)


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_every_figure_builds_for_every_method_and_the_pdf_is_a_pdf(signal):
    for method in C.METHODS:
        a = ev.analyse(ev.Settings(n=256, signal=signal, method=method, noise="an", snr=10.0))
        for fig in (V.build_signal(a), V.build_images(a), V.build_slice(a)):
            assert fig.data
        ridge = V.build_ridge(a)
        assert (ridge is None) == (signal in C.SIGNAL_TWO)
        pdf = build_pdf(a)
        assert pdf[:5] == b"%PDF-" and len(pdf) > 800


def test_display_image_signed_and_unsigned():
    img = np.zeros((4, 512))
    img[:, 10], img[:, 20] = 4.0, -2.0
    z, style = V.display_image(img, signed=True)
    assert z.shape == (4, 307) and z.max() == 1.0 and z.min() == -0.5 and style["zmin"] == -1.0
    z2, style2 = V.display_image(np.abs(img), signed=False)
    assert z2.max() == 0.0 and z2.min() == C.DB_FLOOR and style2["zmin"] == C.DB_FLOOR
    assert (V.display_image(np.zeros((4, 512)), signed=False)[0] == C.DB_FLOOR).all() and (V.display_image(np.zeros((4, 512)), signed=True)[0] == 0).all()
    assert V.power_db(np.array([[1.0, 0.1, 1e-9]]))[0].tolist() == pytest.approx([0.0, -10.0, C.DB_FLOOR])
