"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import rea_evaluation as ev


def make_analysis(signal="Chirp", method="Neu zugeordnet", sigma=12, width=1.1, width_spec=24.9, rmse=0.00002, rmse_spec=0.00028, peaks=2, peaks_spec=2, share=0.9, share_spec=0.3, cross=None, cross_spec=None,
                  neg=0.0):
    z = np.zeros(1)
    two = signal in ("Zwei Töne", "Zwei parallele Chirps")
    s = ev.Settings(signal=signal, method=method, sigma=sigma)
    return ev.Analysis(s, None, z, z, z, z, z, 100, 400, width, width_spec, None if two else rmse, None if two else rmse_spec, peaks, peaks_spec, share, share_spec, cross, cross_spec, neg, 0.0)


def test_main_verdict_spectrogram_is_the_reference():
    kind, text = ev.verdict_main(make_analysis(method="Spektrogramm", width=24.9, rmse=0.00028))
    assert kind == "info"
    assert text == ("Spektrogramm bei σ = 12: wirksame Breite 24,9 Rasterpunkte, Spektrogramm 24,9. Frequenzfehler des Kamms 0,00028. "
                    "Die Breite folgt dem Fenster (1/(2πσ) = 0,0133 in der Frequenz); die Neuzuordnung ist der Vergleich.")
    two = ev.verdict_main(make_analysis(signal="Zwei Töne", method="Spektrogramm", width=48.2, peaks=1))[1]
    assert "Gipfel im Zeitprofil: 1 (wahr: 2)." in two and "Frequenzfehler" not in two
    par = ev.verdict_main(make_analysis(signal="Zwei parallele Chirps", method="Spektrogramm", width=48.2, share=0.283))[1]
    assert "Anteil auf den wahren Linien 28 %." in par and "gegen" not in par.split("Rasterpunkte")[1]


def test_main_verdict_one_component_sharper_worse_and_neutral():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "success"
    assert text == "Neu zugeordnet bei σ = 12: wirksame Breite 1,1 Rasterpunkte, Spektrogramm 24,9. Frequenzfehler des Kamms 0,00002 (Spektrogramm 0,00028)."
    kind, text = ev.verdict_main(make_analysis(rmse=0.0010, rmse_spec=0.0003))
    assert kind == "warning" and text.endswith("Frequenzfehler des Kamms 0,00100 (Spektrogramm 0,00030). Der Gipfel liegt weit neben der Linie: Rauschwerte bestimmen ihn.")
    assert ev.verdict_main(make_analysis(rmse=0.0009, rmse_spec=0.0003))[0] == "success"                  # genau das Dreifache zählt noch nicht (0,0009 < 3 * 0,0003 in Gleitkomma nicht überschritten)
    assert ev.verdict_main(make_analysis(rmse=0.0006, rmse_spec=0.0002))[0] == "success" and ev.verdict_main(make_analysis(rmse=0.00061, rmse_spec=0.0002))[0] == "warning"
    kind, text = ev.verdict_main(make_analysis(width=12.5, width_spec=25.0))
    assert kind == "info" and text.startswith("Neu zugeordnet bei σ = 12: wirksame Breite 12,5 Rasterpunkte, Spektrogramm 25,0.")
    assert ev.verdict_main(make_analysis(width=12.4, width_spec=25.0))[0] == "success"


def test_main_verdict_names_method_and_window():
    assert ev.verdict_main(make_analysis(method="SPWVD", sigma=24, width=3.0))[1].startswith("SPWVD bei σ = 24: wirksame Breite 3,0")


def test_main_verdict_two_tones_cross_term_peaks_and_share():
    kind, text = ev.verdict_main(make_analysis(signal="Zwei Töne", method="Wigner-Ville", width=11.2, width_spec=48.2, cross=1.3, cross_spec=0.0005, neg=-0.999, share=0.33, share_spec=0.28, peaks=3))
    assert kind == "warning"
    assert text == ("Wigner-Ville bei σ = 12: wirksame Breite 11,2 Rasterpunkte, Spektrogramm 48,2. Zwischen den Tönen liegt ein Kreuzterm von 1,30 der Tonhöhe (Spektrogramm 0,0005), mit negativen Werten. "
                    "Anteil auf den wahren Linien 33 % (Spektrogramm 28 %).")
    kind, text = ev.verdict_main(make_analysis(signal="Zwei Töne", method="SPWVD", width=20, width_spec=48.2, cross=0.2, cross_spec=0.0005, neg=-0.01, share=0.33, share_spec=0.28))
    assert kind == "warning" and "Kreuzterm von 0,20 der Tonhöhe (Spektrogramm 0,0005)." in text and "negativen" not in text
    assert ev.verdict_main(make_analysis(signal="Zwei Töne", method="SPWVD", cross=0.2, cross_spec=0.0005, neg=-0.05))[1].count("negativen") == 0
    assert "mit negativen Werten" in ev.verdict_main(make_analysis(signal="Zwei Töne", method="SPWVD", cross=0.2, cross_spec=0.0005, neg=-0.051))[1]
    smooth = ev.verdict_main(make_analysis(signal="Zwei Töne", method="SPWVD", cross=0.1, cross_spec=0.0005, neg=0.0, peaks=2, share=0.9))
    assert smooth[0] == "success"                                                                        # Kreuzterm genau 0,1 zählt noch nicht
    more = ev.verdict_main(make_analysis(signal="Zwei Töne", peaks=6, share=0.45, share_spec=0.32))
    assert more == ("warning", "Neu zugeordnet bei σ = 12: wirksame Breite 1,1 Rasterpunkte, Spektrogramm 24,9. Das Zeitprofil zeigt 6 Gipfel statt 2: es sind Artefakte dabei, die es im Signal nicht gibt. "
                              "Anteil auf den wahren Linien 45 % (Spektrogramm 32 %).")
    less = ev.verdict_main(make_analysis(signal="Zwei Töne", peaks=1, share=0.45, share_spec=0.32))
    assert less[0] == "warning" and "Das Zeitprofil zeigt 1 Gipfel statt 2: die Töne sind nicht getrennt." in less[1]
    ok = ev.verdict_main(make_analysis(signal="Zwei Töne", peaks=2, share=0.97, share_spec=0.28))
    assert ok == ("success", "Neu zugeordnet bei σ = 12: wirksame Breite 1,1 Rasterpunkte, Spektrogramm 24,9. Zwei Gipfel. Anteil auf den wahren Linien 97 % (Spektrogramm 28 %).")
    assert ev.verdict_main(make_analysis(signal="Zwei Töne", peaks=2, share=0.8))[0] == "success" and ev.verdict_main(make_analysis(signal="Zwei Töne", peaks=2, share=0.7999))[0] == "info"


def test_main_verdict_parallel_chirps_use_the_share_not_the_peaks():
    kind, text = ev.verdict_main(make_analysis(signal="Zwei parallele Chirps", peaks=31, share=0.8, share_spec=0.28))
    assert kind == "success" and text.endswith("Rasterpunkte, Spektrogramm 24,9. Anteil auf den wahren Linien 80 % (Spektrogramm 28 %).") and "Gipfel" not in text
    assert ev.verdict_main(make_analysis(signal="Zwei parallele Chirps", method="Synchrosqueezing", peaks=31, share=0.79))[0] == "info"


def _conc(chirp=(48.3, 24.9, 17.6, 1.1, 1.9, 5.7, 11.2), fm=(51.6, 40.7, 39.7, 3.6, 11.4, 21.8, 31.2), tone=(48.2, 24.1, 12.1, 1.0, 1.0, 5.6, 10.2)):
    rows = []
    for name, v in (("Chirp", chirp), ("Sinus-FM", fm), ("Ton", tone)):
        rows.append({"signal": name, "spec": {6.0: v[0], 12.0: v[1], 24.0: v[2]}, "reassigned": v[3], "squeezed": v[4], "wvd": v[5], "spwvd": v[6]})
    return rows


def test_verdict_concentration_text_and_kind():
    kind, text = ev.verdict_concentration(_conc())
    assert kind == "success"
    assert text == ("Beim Chirp schrumpft die wirksame Breite von 24,9 (Spektrogramm, σ = 12) auf 1,1 (Neu zuordnen) und 1,9 (Synchrosqueezing); Wigner-Ville 5,7. "
                    "Bei der Sinus-FM bleibt mehr übrig: 40,7 auf 3,6 und 11,4 (Wigner-Ville 21,8), vermutlich, weil die Linie sich innerhalb des Fensters krümmt.")
    assert ev.verdict_concentration(_conc(tone=(48.2, 24.1, 12.1, 24.1, 1.0, 5.6, 10.2)))[0] == "warning" and ev.verdict_concentration(_conc(tone=(48.2, 24.1, 12.1, 24.0, 1.0, 5.6, 10.2)))[0] == "success"


def _cross(smooth):
    return {"f1": 0.1, "f2": 0.18, "period": 12.5, "negative": -2.0163,
            "methods": [{"method": "Spektrogramm", "ratio": 0.00049, "width": 48.2}, {"method": "Neu zugeordnet", "ratio": 4e-6, "width": 2.0}, {"method": "Synchrosqueezing", "ratio": 1e-7, "width": 2.0},
                        {"method": "Wigner-Ville", "ratio": 1.298, "width": 11.2}],
            "smooth": smooth}


def test_verdict_cross_text():
    smooth = [{"sigma_t": 0.0, "sigma_lag": 20.0, "ratio": 1.27, "width": 24.1}, {"sigma_t": 6.0, "sigma_lag": 20.0, "ratio": 0.0134, "width": 20.6}, {"sigma_t": 12.0, "sigma_lag": 20.0, "ratio": 0.0, "width": 20.5}]
    kind, text = ev.verdict_cross(_cross(smooth))
    assert kind == "warning"
    assert text == ("Die Wigner-Ville-Verteilung zeigt zwischen den Tönen einen Kreuzterm von 1,30 der Tonhöhe, mit Werten bis −2,0 der Tonhöhe; das Spektrogramm hat dort 0,0005. "
                    "Der Kreuzterm schwingt in der Zeit mit der Periode 1/(f₂ − f₁) = 12,5 Abtastwerten. Eine Glättung in der Zeit (σ_t = 6, Verzögerungsfenster 20) drückt ihn auf 0,013, die Töne werden dafür breiter (20,6 Rasterpunkte).")
    none = ev.verdict_cross(_cross(smooth[:1]))[1]
    assert none.endswith("12,5 Abtastwerten.") and "Glättung" not in none
    edge = ev.verdict_cross(_cross([{"sigma_t": 6.0, "sigma_lag": 20.0, "ratio": 0.1, "width": 20.6}]))[1]
    assert "Glättung" not in edge


def _noise(sst0=0.054, re0=0.0016, sp0=0.0017):
    rows = []
    for snr, sp, re, sst, gate in ((60, 0.00028, 0.00002, 0.00028, 0.00028), (0, sp0, re0, sst0, 0.0068), (-5, 0.067, 0.067, 0.137, 0.106)):
        rows.append({"snr": snr, "runs": 20, "Spektrogramm": sp, "Neu zugeordnet": re, "Synchrosqueezing": sst, "Synchrosqueezing mit Schwelle": gate, "Wigner-Ville": 0.001, "SPWVD": 0.0008})
    return rows


def test_verdict_noise_text_and_kind():
    kind, text = ev.verdict_noise(_noise())
    assert kind == "warning"
    assert text == ("Bei hohem Rauschabstand (60 dB) ist die neu zugeordnete Frequenz am Gipfel genauer als der Gipfel des Spektrogramms (0,00002 gegen 0,00028, das Raster gibt 0,00028 vor); "
                    "bei 0 dB ist der Vorsprung weg (0,0016 gegen 0,0017). Synchrosqueezing bricht bei 0 dB ein (0,0540); mit einer Schwelle von 3 Effektivwerten des Rauschens in der STFT sind es 0,0068.")
    assert ev.verdict_noise(_noise(sst0=0.0085))[0] == "warning" and ev.verdict_noise(_noise(sst0=0.0085 - 1e-9))[0] == "info"        # 5-faches des Spektrogramms (0,0017)
    assert "ist der Vorsprung kleiner (0,0010 gegen 0,0017)" in ev.verdict_noise(_noise(re0=0.001))[1]
    assert "ist der Vorsprung weg (0,0017 gegen 0,0018)" in ev.verdict_noise(_noise(re0=0.0017, sp0=0.0018))[1] and "ist der Vorsprung kleiner (0,0016 gegen 0,0018)" in ev.verdict_noise(_noise(sp0=0.0018))[1]
    assert "Vorsprung kleiner" in ev.verdict_noise(_noise(re0=0.0018, sp0=0.002))[1] and "Vorsprung weg" in ev.verdict_noise(_noise(re0=0.00181, sp0=0.002))[1]


def _res(rows):
    return [{"df": df, "widths": df * 2 * np.pi * 12, "peaks_spec": ps, "peaks_re": pr, "share_spec": ss, "share_re": sr} for df, ps, pr, ss, sr in rows]


def test_verdict_resolution_text():
    rows = _res([(0.01, 1, 1, 0.45, 0.44), (0.02, 2, 6, 0.32, 0.50), (0.05, 2, 2, 0.28, 0.97)])
    kind, text = ev.verdict_resolution(rows)
    assert kind == "warning"
    assert text == ("Das Spektrogramm zeigt zwei Gipfel ab df = 0,020 (1,5 Fensterbreiten). Das neu zugeordnete Bild zeigt bei df = 0,020 6 Gipfel statt 2: Artefakte, kein Gewinn an Auflösung. "
                    "Erst bei df = 0,050 liegen 97 % der neu zugeordneten Leistung auf den wahren Linien (Spektrogramm 28 %).")
    clean = ev.verdict_resolution(_res([(0.01, 1, 1, 0.45, 0.44), (0.05, 2, 2, 0.28, 0.97)]))[1]
    assert "Artefakte" not in clean and clean.startswith("Das Spektrogramm zeigt zwei Gipfel ab df = 0,050")
    none = ev.verdict_resolution(_res([(0.01, 1, 1, 0.45, 0.44)]))[1]
    assert none.startswith("Erst bei df = 0,010")


def _win(rows):
    return [{"sigma": s, "runs": 20, "snr": 10, "clean": {"Spektrogramm": sp, "Neu zugeordnet": re, "Neu zugeordnet (zur neuen Zeit)": nz, "Synchrosqueezing": 0.02}, "noisy": {}} for s, sp, re, nz in rows]


def test_verdict_window_text():
    kind, text = ev.verdict_window(_win([(4, 0.0009, 0.00086, 0.00087), (32, 0.01322, 0.01319, 0.00854)]))
    assert kind == "info"
    assert text == ("Der Frequenzfehler des Gipfels wächst mit der Fensterbreite: 0,00090 bei σ = 4, 0,01322 bei σ = 32. Die neu zugeordnete Frequenz am Gipfel ändert daran höchstens 4,4 %; "
                    "erst gegen die Frequenz zur neu zugeordneten Zeit sinkt der Fehler bei σ = 32 auf 0,00854.")
    assert "höchstens 10,0 %" in ev.verdict_window(_win([(4, 0.001, 0.0011, 0.001), (32, 0.01, 0.01, 0.01)]))[1]


def test_verdict_cross_picks_the_first_time_smoothed_row_below_a_tenth():
    base = {"sigma_lag": 20.0, "width": 20.0}
    rows = [dict(base, sigma_t=0.0, ratio=0.05), dict(base, sigma_t=1.0, ratio=0.04), dict(base, sigma_t=6.0, ratio=0.0134)]
    assert "(σ_t = 1, Verzögerungsfenster 20) drückt ihn auf 0,040" in ev.verdict_cross(_cross(rows))[1]               # σ_t = 0 zählt nicht als Zeitglättung
    rows2 = [dict(base, sigma_t=0.0, ratio=0.05), dict(base, sigma_t=6.0, ratio=0.0134)]
    assert "(σ_t = 6, Verzögerungsfenster 20)" in ev.verdict_cross(_cross(rows2))[1]
    assert "Glättung" not in ev.verdict_cross(_cross([dict(base, sigma_t=0.0, ratio=0.05)]))[1]


def test_verdict_noise_thresholds_at_exact_equality():
    assert ev.verdict_noise(_noise(sst0=2.5, sp0=0.5, re0=0.1))[0] == "info"                                       # genau das Fünffache zählt nicht
    assert ev.verdict_noise(_noise(sst0=2.6, sp0=0.5, re0=0.1))[0] == "warning"
    assert "Vorsprung kleiner" in ev.verdict_noise(_noise(re0=0.9, sp0=1.0))[1] and "Vorsprung weg" in ev.verdict_noise(_noise(re0=0.91, sp0=1.0))[1]


def test_verdict_resolution_names_the_first_row_with_more_than_two_peaks():
    rows = _res([(0.01, 1, 1, 0.45, 0.44), (0.02, 2, 3, 0.32, 0.50), (0.05, 2, 2, 0.28, 0.97)])
    assert "bei df = 0,020 3 Gipfel statt 2: Artefakte" in ev.verdict_resolution(rows)[1]
    assert "Artefakte" not in ev.verdict_resolution(_res([(0.02, 2, 2, 0.32, 0.50)]))[1]
