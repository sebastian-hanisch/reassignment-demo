"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Grenzen, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import math
from pathlib import Path

import numpy as np
import pytest

import rea_constants as C
import rea_evaluation as ev
import rea_measure as M
import rea_signals as SG
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
de = ev.de


def preset(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], sigma=p["sigma"], method=p["method"], df=p["df"], noise=p["noise"], snr=p["snr"], sigma_t=p["sigma_t"], sigma_lag=p["sigma_lag"], floor=p["floor"], gate=p["gate"],
                seed=p["seed"])
    return ev.analyse(ev.Settings(**{**base, **over}))


P = list(C.PRESETS)


def rows_by(key, name):
    return {r[key]: r for r in get(name)}


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_chirp_about_25_points_to_about_one():
    a = preset(P[0])
    assert 24 < a.width_spec < 26 and 1.0 <= a.width < 1.5 and "etwa 25 Rasterpunkte" in C.PRESET_HELP[P[0]] and "auf etwa einen zusammen" in C.PRESET_HELP[P[0]]
    assert "0,05 bis 0,20" in C.PRESET_HELP[P[0]] and (C.CHIRP_F0, C.CHIRP_F1) == (0.05, 0.20) and "σ = 12" in C.PRESET_HELP[P[0]] and a.settings.sigma == 12


def test_preset_sine_fm_squeezing_is_less_sharp_than_reassignment_and_a_rest_remains():
    a = preset(P[1])
    re = preset(P[1], method="Neu zugeordnet")
    assert a.settings.method == "Synchrosqueezing" and re.width < a.width < a.width_spec and "schärft stark, Synchrosqueezing" in C.PRESET_HELP[P[1]] and "weniger" in C.PRESET_HELP[P[1]]
    assert re.width > 1.5                                         # an den Umkehrpunkten der Linie bleibt ein Rest an Breite


def test_preset_two_tones_wigner_ville_cross_term_as_large_as_the_tones_and_negative():
    a = preset(P[2])
    assert 1.0 < a.cross < 1.5 and a.neg < -0.5 and a.cross_spec < 0.001 and (C.TWO_F1, C.TWO_F1 + a.settings.df) == (0.10, pytest.approx(0.18))
    h = C.PRESET_HELP[P[2]]
    assert "bei 0,10 und 0,18" in h and "von der Größe der Töne selbst" in h and "mit negativen Werten" in h and "fast nichts" in h


def test_preset_close_tones_the_spectrogram_just_separates_them_and_reassignment_adds_peaks():
    a = preset(P[3])
    h = C.PRESET_HELP[P[3]]
    assert a.settings.df == 0.02 and de(a.settings.df * 2 * math.pi * a.settings.sigma, 1) == "1,5" and "Abstand 0,02 (1,5 Fensterbreiten)" in h
    assert a.peaks_spec == 2 and a.peaks > 2 and "gerade noch" in h and "zusätzliche Gipfel" in h


def test_preset_noise_squeezing_outlier_and_the_gate_removes_it():
    a, g = preset(P[4]), preset(P[4], gate="an")
    h = C.PRESET_HELP[P[4]]
    assert a.settings.snr == 0.0 and de(a.rmse, 2) == "0,08" and de(a.rmse_spec, 4) == "0,0015" and "Fehler 0,08 gegen 0,0015" in h
    assert g.rmse < 2 * g.rmse_spec and g.rmse < 0.01 and "schalten Sie die Rauschschwelle ein" in h
    row = rows_by("snr", "noise")[0]
    assert row["Synchrosqueezing mit Schwelle"] > row["Spektrogramm"] and "Im Mittel über 20 Läufe" in h and row["runs"] == 20 and "schlechter als das Spektrogramm" in h


# --- README: Befunde --------------------------------------------------------------------------------------------------------------------------


def test_readme_sharpness():
    rows = rows_by("signal", "concentration")
    ch, fm, tone = rows["Chirp"], rows["Sinus-FM"], rows["Ton"]
    assert (f"Beim linearen Chirp sinkt die wirksame Breite von {de(ch['spec'][12.0])} Rasterpunkten (Spektrogramm, σ = 12) auf {de(ch['reassigned'])} (Neu zuordnen) und {de(ch['squeezed'])} (Synchrosqueezing); "
            f"Wigner-Ville {de(ch['wvd'])}, SPWVD {de(ch['spwvd'])}.") in README
    assert f"Bei der Sinus-FM von {de(fm['spec'][12.0])} auf {de(fm['reassigned'])} und {de(fm['squeezed'])} (Wigner-Ville {de(fm['wvd'])}, SPWVD {de(fm['spwvd'])})" in README
    assert f"Beim Ton von {de(tone['spec'][12.0])} auf {de(tone['reassigned'])} und {de(tone['squeezed'])} (Wigner-Ville {de(tone['wvd'])})" in README
    assert (ch["reassigned"] < ch["spec"][12.0] / 10) and fm["reassigned"] < fm["spec"][12.0] / 10 and tone["reassigned"] == pytest.approx(1.0, abs=1e-3)


def test_readme_cross_terms():
    r = get("cross")
    by = {m["method"]: m for m in r["methods"]}
    sm = {(x["sigma_t"], x["sigma_lag"]): x for x in r["smooth"]}
    assert f"einen Kreuzterm von {de(by['Wigner-Ville']['ratio'], 2)} der Tonhöhe, mit Werten bis {de(r['negative'])} der Tonhöhe; das Spektrogramm hat dort {de(by['Spektrogramm']['ratio'], 4)}" in README
    assert "−2,0 der Tonhöhe" in README and "Spektrogramm hat dort 0,0005" in README and by["Neu zugeordnet"]["ratio"] < by["Spektrogramm"]["ratio"] and by["Synchrosqueezing"]["ratio"] < by["Spektrogramm"]["ratio"]
    assert f"Periode 1/(f₂ − f₁) = {de(r['period'])} Abtastwerten" in README and r["period"] == pytest.approx(1 / 0.08)
    no_time = [sm[(0.0, lag)]["ratio"] for lag in (255.0, 40.0, 20.0)]
    assert f"(σ_t = 0) lässt ihn stehen ({de(min(no_time), 2)} bis {de(max(no_time), 2)})" in README
    assert f"σ_t = 6 drückt ihn auf {de(sm[(6.0, 20.0)]['ratio'], 3)} (Breite der Töne {de(sm[(6.0, 20.0)]['width'])} Punkte), σ_t = 12 auf unter 0,001" in README and sm[(12.0, 20.0)]["ratio"] < 0.001


def test_readme_noise():
    rows = rows_by("snr", "noise")
    hi, lo, bad = rows[60], rows[0], rows[-5]
    assert f"Bei 60 dB liefert die neu zugeordnete Frequenz am Gipfel {de(hi['Neu zugeordnet'], 5)} gegen {de(hi['Spektrogramm'], 5)} beim Gipfel des Spektrogramms (das Raster gibt {de(1 / (math.sqrt(12) * 1024), 5)} vor)" in README
    assert f"Bei 0 dB gibt es keinen Vorsprung mehr: {de(lo['Neu zugeordnet'], 4)} gegen {de(lo['Spektrogramm'], 4)}" in README and lo["Neu zugeordnet"] > 0.9 * lo["Spektrogramm"]
    assert f"Synchrosqueezing bricht bei 0 dB ein ({de(lo['Synchrosqueezing'], 4)}); mit einer Schwelle von 3 Effektivwerten des Rauschens in der STFT sind es {de(lo['Synchrosqueezing mit Schwelle'], 4)}, immer noch schlechter" in README
    assert lo["Synchrosqueezing mit Schwelle"] > lo["Spektrogramm"] and lo["Synchrosqueezing"] > 20 * lo["Spektrogramm"]
    assert f"Wigner-Ville ({de(lo['Wigner-Ville'], 4)}) und SPWVD ({de(lo['SPWVD'], 4)}) sind bei 0 dB besser als das Spektrogramm" in README and max(lo["Wigner-Ville"], lo["SPWVD"]) < lo["Spektrogramm"]
    values = [bad[m] for m in ev.NOISE_METHODS]
    assert f"Bei −5 dB versagen alle ({de(min(values), 2)} bis {de(max(values), 2)})" in README and min(values) > 0.02 and (de(min(values), 2), de(max(values), 2)) == ("0,03", "0,14")


def test_readme_resolution():
    rows = rows_by("df", "resolution")
    assert f"Das Spektrogramm zeigt zwei Gipfel ab df = {de(0.02, 3)} ({de(rows[0.02]['widths'])} Fensterbreiten)" in README
    assert rows[0.01]["peaks_spec"] == rows[0.015]["peaks_spec"] == 1 and rows[0.02]["peaks_spec"] == 2 and rows[0.02]["peaks_re"] == 6 and "bei dieser Distanz sechs" in README
    assert rows[0.01]["peaks_re"] == 1 and rows[0.01]["widths"] < 1.0 and "unter einer Fensterbreite wie das Spektrogramm einen" in README
    assert rows[0.03]["peaks_re"] == 2 and rows[0.03]["widths"] > 2.0 and "Erst ab zwei Fensterbreiten (df = 0,030) sind es wieder zwei" in README
    assert f"bei df = 0,050 liegen {de(100 * rows[0.05]['share_re'], 0)} % der neu zugeordneten Leistung auf den wahren Linien (Spektrogramm {de(100 * rows[0.05]['share_spec'], 0)} %)" in README


def test_readme_window():
    rows = rows_by("sigma", "window")
    c = {s: r["clean"] for s, r in rows.items()}
    assert (f"{de(c[4]['Spektrogramm'], 5)} bei σ = 4, {de(c[12]['Spektrogramm'], 5)} bei σ = 12, {de(c[32]['Spektrogramm'], 5)} bei σ = 32") in README
    assert f"nichts ({de(c[12]['Neu zugeordnet'], 5)} bei σ = 12)" in README and abs(c[12]["Neu zugeordnet"] - c[12]["Spektrogramm"]) < 0.01 * c[12]["Spektrogramm"]
    assert f"bei σ = 32 auf {de(c[32]['Neu zugeordnet (zur neuen Zeit)'], 5)}" in README and c[32]["Neu zugeordnet (zur neuen Zeit)"] < 0.7 * c[32]["Spektrogramm"]
    assert c[4]["Spektrogramm"] < c[12]["Spektrogramm"] < c[32]["Spektrogramm"]


# --- README: Methodik, Modell, Grenzen ---------------------------------------------------------------------------------------------------------


def test_readme_methodology_numbers():
    assert M.inner(512) == (100, 412) and "Zeiten 100 bis 411" in README and M.ridge_window(512) == (60, 452) and "Zeiten 60 bis 451" in README
    assert C.EXP_N == 512 and C.EXP_SIGMA == 12.0 and f"Breite in der Frequenz 1/(2πσ) = {de(ev.window_width(12.0), 4)}" in README and C.EXP_SEED == 1700 and "Seed 1700" in README
    assert C.EXP_NOISE_RUNS == C.EXP_WINDOW_RUNS == 20 and "20 Läufe" in README and C.NFFT == 1024 and "Frequenzraster 1/1024" in README and C.BAND == 3 and "± 3 Rasterpunkten" in README
    assert (min(C.N_CHOICES), max(C.N_CHOICES)) == (256, 1024) and (C.SIGMA_MIN, C.SIGMA_MAX) == (4, 32) and "n = 256 bis 1024" in README and "σ = 4 bis 32" in README
    assert C.GATE_FACTOR == 3.0 and "3 Effektivwerten" in README and 2 * 255 + 1 <= 512 and "Verzögerungen bis ±255" in README


def test_readme_model_signals():
    assert (C.TONE_F, C.CHIRP_F0, C.CHIRP_F1, C.SINFM_FC, C.SINFM_DF, C.SINFM_FM, C.TWO_F1, C.PAR_F0, C.PAR_F1) == (0.137, 0.05, 0.20, 0.12, 0.04, 0.01, 0.10, 0.04, 0.10)
    assert ("Ton 0,137, linearer Chirp 0,05 bis 0,20, Sinus-FM mit Mitte 0,12, Hub 0,04 und Modulation 0,01, zwei Töne 0,10 und 0,10 + df, zwei parallele Chirps 0,04 bis 0,10 im Abstand df") in README
    for kind in C.SIGNALS:
        sig = SG.make_signal(kind, 512, C.DF_MAX)
        assert max(t.max() for t in sig.tracks) < 0.25
    assert "unter 0,25" in README


def test_readme_limit_about_the_analytic_signal_deviation():
    sig = SG.make_signal("Sinus-FM", 512)
    ph = np.unwrap(np.angle(SG.analytic(sig.x)))
    dev = np.abs((ph[2:] - ph[:-2]) / (4 * np.pi) - sig.tracks[0][1:-1])[30:-30].max()
    assert 0.002 < dev < 0.003 and "um bis zu 0,003" in README


def test_readme_dev_dependency_and_footer_sections():
    assert "scipy" in (ROOT / "requirements-dev.txt").read_text() and "scipy" not in (ROOT / "requirements.txt").read_text()
    for section in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt"):
        assert section in README
    for f in ROOT.glob("rea_*.py"):
        assert f"`{f.name}`" in README, f.name


# --- App-Texte --------------------------------------------------------------------------------------------------------------------------------


def test_app_texts():
    assert f"(bei σ = 12 etwa {de(ev.window_width(12.0), 3)})" in APP and "Die Fensterbreite in der Frequenz ist 1/(2πσ)" in APP
    rows = rows_by("snr", "noise")
    assert "bricht ab etwa 0 dB ein" in APP and rows[0]["Synchrosqueezing"] > 5 * rows[0]["Spektrogramm"] and rows[10]["Synchrosqueezing"] < 2 * rows[10]["Spektrogramm"]
    assert "alle Verfahren versagen bei etwa −5 dB" in APP and min(rows[-5][m] for m in ev.NOISE_METHODS) > 0.02
    res = rows_by("df", "resolution")
    assert "Zwei Töne unter etwa einer Fensterbreite bleiben einer, knapp darüber entstehen zusätzliche Gipfel" in APP and res[0.01]["peaks_re"] == 1 and res[0.02]["peaks_re"] > 2
    assert "Ziel 0,05 bis 0,20" not in APP and "lineare Frequenzrampe 0,05 bis 0,20" in APP and "Kreuzterme von der Größe der Komponenten" in APP
    assert "das Raster gibt 0,00028 vor" in APP and de(1 / (math.sqrt(12) * C.NFFT), 5) == "0,00028"
    assert "Zeiten 100 bis 411" in APP and "Dreifache des Effektivwerts" in APP and "Frequenzen über 0,25 würden bei der Wigner-Ville-Verteilung falten" in APP
    assert "Stück 14" in APP and "vierzehnte Demo" in APP
