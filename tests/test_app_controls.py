"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import rea_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
DF = "Abstand der Komponenten df"
SNR = "Rauschabstand (dB)"
FLOOR = "Leistungsschwelle (dB gegen die Spitze)"
SIGMA_T = "SPWVD: Glättung in der Zeit σ_t"
SIGMA_LAG = "SPWVD: Verzögerungsfenster σ_lag"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


def _sliders(at):
    return {x.label: x for x in at.slider}


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_every_signal_runs(signal):
    at = _run()
    at.selectbox(key="signal_select").set_value(signal).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("method", list(C.METHODS))
def test_every_method_runs_with_and_without_noise(method):
    at = _run({"m": method})
    assert not at.exception
    at.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_distance_slider_exists_only_for_two_component_signals_and_keeps_its_value():
    at = _run()
    assert DF not in _labels(at)
    at.selectbox(key="signal_select").set_value("Zwei Töne").run(timeout=TIMEOUT)
    assert DF in _labels(at) and _sliders(at)[DF].value == C.DEFAULT_DF
    _sliders(at)[DF].set_value(0.03).run(timeout=TIMEOUT)
    at.selectbox(key="signal_select").set_value("Chirp").run(timeout=TIMEOUT)
    at.selectbox(key="signal_select").set_value("Zwei parallele Chirps").run(timeout=TIMEOUT)
    assert not at.exception and _sliders(at)[DF].value == 0.03


def test_the_noise_controls_follow_the_noise_switch_and_the_method():
    at = _run()
    assert SNR not in _labels(at) and "Rauschschwelle" not in [r.label for r in at.radio]
    at.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert SNR in _labels(at) and "Rauschschwelle" in [r.label for r in at.radio]          # Standardverfahren: Neu zugeordnet
    at.radio(key="method_select").set_value("Wigner-Ville").run(timeout=TIMEOUT)
    assert SNR in _labels(at) and "Rauschschwelle" not in [r.label for r in at.radio] and FLOOR not in _labels(at)
    at.radio(key="method_select").set_value("Synchrosqueezing").run(timeout=TIMEOUT)
    assert FLOOR in _labels(at) and "Rauschschwelle" in [r.label for r in at.radio]


def test_the_spwvd_sliders_exist_only_for_the_spwvd():
    at = _run()
    assert SIGMA_T not in _labels(at) and SIGMA_LAG not in _labels(at)
    at.radio(key="method_select").set_value("SPWVD").run(timeout=TIMEOUT)
    assert not at.exception and SIGMA_T in _labels(at) and SIGMA_LAG in _labels(at) and FLOOR not in _labels(at)
    _sliders(at)[SIGMA_T].set_value(12).run(timeout=TIMEOUT)
    at.radio(key="method_select").set_value("Neu zugeordnet").run(timeout=TIMEOUT)
    at.radio(key="method_select").set_value("SPWVD").run(timeout=TIMEOUT)
    assert _sliders(at)[SIGMA_T].value == 12


def test_metric_columns_follow_the_signal():
    at = _run()
    assert [m.label for m in at.metric] == ["Wirksame Breite", "Frequenzfehler", "Anteil auf den Linien", "Kleinster / größter Wert"]
    at.selectbox(key="signal_select").set_value("Zwei Töne").run(timeout=TIMEOUT)
    assert [m.label for m in at.metric] == ["Wirksame Breite", "Gipfel im Zeitprofil", "Anteil auf den Linien", "Kleinster / größter Wert", "Kreuzterm / Ton"]
    at.selectbox(key="signal_select").set_value("Zwei parallele Chirps").run(timeout=TIMEOUT)
    assert [m.label for m in at.metric] == ["Wirksame Breite", "Anteil auf den Linien", "Kleinster / größter Wert"]


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "1024", "s": "Zwei Töne", "sg": "20", "m": "SPWVD", "df": "0.05", "rs": "an", "snr": "3.5", "st": "9", "sl": "30", "fl": "-40", "gt": "an", "seed": "42"})
    assert at.select_slider(key="n_select").value == 1024 and at.selectbox(key="signal_select").value == "Zwei Töne" and at.radio(key="method_select").value == "SPWVD" and at.radio(key="noise_select").value == "an"
    s = _sliders(at)
    assert s["Fensterbreite σ (Abtastwerte)"].value == 20 and s[DF].value == 0.05 and s[SNR].value == 3.5 and s[SIGMA_T].value == 9 and s[SIGMA_LAG].value == 30 and at.number_input(key="seed_input").value == 42
    at2 = _run({"n": "3000", "s": "unsinn", "sg": "99", "m": "Hough", "df": "5", "rs": "vielleicht", "snr": "99", "st": "-4", "sl": "999", "fl": "7", "gt": "x"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="signal_select").value == C.DEFAULT_SIGNAL and at2.radio(key="method_select").value == C.DEFAULT_METHOD
    assert at2.radio(key="noise_select").value == C.DEFAULT_NOISE and _sliders(at2)["Fensterbreite σ (Abtastwerte)"].value == C.SIGMA_MAX
    at3 = _run({"s": "Zwei Töne", "df": "5", "rs": "an", "snr": "99"})
    assert _sliders(at3)[DF].value == C.DF_MAX and _sliders(at3)[SNR].value == C.SNR_MAX


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key=f"preset_{list(C.PRESETS)[3]}").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="signal_select").value == "Zwei Töne" and _sliders(at)[DF].value == 0.02 and at.radio(key="method_select").value == "Neu zugeordnet"
    at.button(key=f"preset_{list(C.PRESETS)[4]}").click().run(timeout=TIMEOUT)
    assert at.radio(key="noise_select").value == "an" and _sliders(at)[SNR].value == 0.0 and at.radio(key="method_select").value == "Synchrosqueezing" and at.selectbox(key="signal_select").value == "Chirp"


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
