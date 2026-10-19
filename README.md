# Neu zuordnen statt verschmieren: Reassignment und Synchrosqueezing schärfen das Spektrogramm – und wo die Wigner-Ville-Verteilung scheitert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-reassignment-demo.streamlit.app/)**

Vierzehnte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos zum **Spektrogramm** und zum **Skalogramm**.
Ein Spektrogramm mit Gauß-Fenster wird auf den Energieschwerpunkt seiner Punkte **neu zugeordnet** (Reassignment), nur in der Frequenz verschoben (**Synchrosqueezing**) und mit der **Wigner-Ville-Verteilung** und ihrer
geglätteten Pseudo-Form (**SPWVD**, Cohen-Klasse) verglichen. Gemessen werden Schärfe, Kreuzterme, Frequenzschätzung im Rauschen, die Trennung zweier Töne und die Fensterbreite, jeweils gegen die bekannte wahre Frequenz.

## Warum dieses Problem

Ein Spektrogramm verschmiert eine Linie über die Breite des Fensters; dahinter steht die Unschärferelation zwischen Zeit- und Frequenzauflösung. Reassignment und Synchrosqueezing schärfen das Bild nachträglich, ohne das Fenster
zu ändern, die Wigner-Ville-Verteilung braucht mathematisch kein Fenster (die Implementierung schneidet die Verzögerung bei ±255 ab), hat aber Kreuzterme. Die Demo misst, was davon bleibt: wie viel schärfer, ob die Auflösung zweier Töne besser wird, was Rauschen anrichtet.

## Modell

Reelle Testsignale (Ton 0,137, linearer Chirp 0,05 bis 0,20, Sinus-FM mit Mitte 0,12, Hub 0,04 und Modulation 0,01, zwei Töne 0,10 und 0,10 + df, zwei parallele Chirps 0,04 bis 0,10 im Abstand df), Amplitude 1 je Komponente,
Frequenzen in Zyklen je Abtastwert und unter 0,25 (konservative Wahl: die Wigner-Ville-Verteilung des analytischen Signals faltet in dieser Formulierung erst am Rand 0 / 0,5). Das analytische Signal entsteht per FFT. STFT mit Gauß-Fenster der Standardabweichung σ, seiner Ableitung und
seiner Zeitgewichtung; Schwerpunkt t̂ = t + Re(X_tg/X_g), f̂ = f − Im(X_g′/X_g)/(2π); Neuzuordnung sammelt |X|² am nächsten Rasterpunkt von (t̂, f̂), Synchrosqueezing die komplexen Werte nur nach f̂. Wigner-Ville
W(t, f) = Σ_τ z[t+τ]·conj(z[t−τ])·e^(−i4πfτ) mit Verzögerungen bis ±255; SPWVD mit Gauß-Fenster in der Verzögerung (σ_lag) und Gauß-Glättung in der Zeit (σ_t). Rauschen: weißes Gauß-Rauschen auf dem reellen Signal, der
Rauschabstand gilt gegen eine Linie der Amplitude 1. Alles in numpy; scipy dient nur als Gegenprobe in den Tests.

## Methodik

n = 256 bis 1024 (Experimente: 512), Frequenzraster 1/1024, Gauß-Fenster σ = 4 bis 32 (Experimente: 12, Breite in der Frequenz 1/(2πσ) = 0,0133). Die **wirksame Breite** ist die Summe der Beträge gegen das Maximum je Zeitpunkt
(Rasterpunkte; 1 ist eine scharfe Linie), gemittelt über die Zeiten 100 bis 411. Der **Frequenzfehler** ist der quadratische Mittelwert der Abweichung des Gipfels von der wahren Frequenz (Zeiten 60 bis 451); in den Experimenten ist es das Mittel dieser Werte über die Läufe. Der **Anteil auf den
wahren Linien** zählt die Beträge innerhalb von ± 3 Rasterpunkten einer wahren Linie. Experimente über feste Zufallsströme (Seed 1700): 20 Läufe je Rauschabstand und je Fensterbreite.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie viel schärfer wird das Spektrogramm (Experiment *Schärfe*)? | Beim linearen Chirp sinkt die wirksame Breite von 24,9 Rasterpunkten (Spektrogramm, σ = 12) auf 1,1 (Neu zuordnen) und 1,9 (Synchrosqueezing); Wigner-Ville 5,7, SPWVD 11,2. Bei der Sinus-FM von 40,7 auf 3,6 und 11,4 (Wigner-Ville 21,8, SPWVD 31,2), vermutlich weil sich die Linie im Fenster krümmt (nicht isoliert gemessen). Beim Ton von 24,1 auf 1,0 und 1,0 (Wigner-Ville 5,6). |
| Was sind Kreuzterme (*Kreuzterme*)? | Bei zwei gleich starken Tönen (0,10 und 0,18) zeigt die Wigner-Ville-Verteilung in der Mitte einen Kreuzterm von 1,30 der Tonhöhe, mit Werten bis −2,0 der Tonhöhe; das Spektrogramm hat dort 0,0005, Neu zuordnen und Synchrosqueezing noch weniger. Der Kreuzterm schwingt mit der Periode 1/(f₂ − f₁) = 12,5 Abtastwerten. Ein Verzögerungsfenster allein (σ_t = 0) lässt ihn stehen (1,27 bis 1,30); eine Glättung in der Zeit mit σ_t = 6 drückt ihn auf 0,013 (Breite der Töne 20,6 Punkte), σ_t = 12 auf unter 0,001. |
| Was richtet Rauschen an (*Frequenzschätzung im Rauschen*)? | Bei 60 dB liefert die neu zugeordnete Frequenz am Gipfel 0,00002 gegen 0,00028 beim Gipfel des Spektrogramms (das Raster gibt 0,00028 vor). Bei 0 dB gibt es keinen Vorsprung mehr: 0,0016 gegen 0,0017. Synchrosqueezing bricht bei 0 dB ein (0,0540); mit einer Schwelle von 3 Effektivwerten des Rauschens in der STFT sind es 0,0068, immer noch schlechter. Wigner-Ville (0,0010) und SPWVD (0,0008) liegen bei 0 dB im Mittel der Lauf-RMSE unter dem Spektrogramm (Wigner-Ville streut stark). Bei −5 dB liegen Spektrogramm, Neu zuordnen, Wigner-Ville und Synchrosqueezing bei 0,06 bis 0,14, nur die SPWVD bleibt bei 0,03. |
| Löst Neu zuordnen zwei nahe Töne besser auf (*Zwei Töne trennen*)? | Nein. Das Spektrogramm zeigt zwei Gipfel ab df = 0,020 (1,5 Fensterbreiten); das neu zugeordnete Bild zeigt bei df = 0,015 zwei Gipfel an falschen Lagen (Artefakte) und bei df = 0,020 sechs, unter einer Fensterbreite zeigt es wie das Spektrogramm einen. Erst ab df = 0,030 (mehr als zwei Fensterbreiten) sind es wieder zwei; bei df = 0,050 liegen 97 % der neu zugeordneten Leistung auf den wahren Linien (Spektrogramm 28 %). Das Band ist ± 3 Rasterpunkte; die 28 % messen die Verschmierung des Spektrogramms, nicht eine falsche Lage. Gilt für gleich starke Töne bei σ = 12. |
| Welche Rolle spielt die Fensterbreite (*Fensterbreite*)? | Der Frequenzfehler des Gipfels bei der Sinus-FM wächst mit σ: 0,00090 bei σ = 4, 0,00454 bei σ = 12, 0,01322 bei σ = 32. Die neu zugeordnete Frequenz am Gipfel ändert daran höchstens um 4,5 % (0,00453 bei σ = 12); erst gegen die wahre Frequenz zur neu zugeordneten Zeit sinkt der Fehler bei σ = 32 auf 0,00854. |

## Ehrliche Grenzen

- **Heimvorteil:** Ton, Chirp, Sinus-FM und zwei Töne sind einkomponentige oder getrennte, glatte Signale, für die Neu zuordnen gebaut ist. Impulse, Mehrkomponentensignale mit Überlappung und Amplitudenmodulation sind nicht gemessen.
- **Die wirksame Breite misst Schärfe, nicht Richtigkeit:** Eine scharfe, aber falsch platzierte Linie hat eine kleine Breite; deshalb stehen daneben der Frequenzfehler und der Anteil auf den wahren Linien.
- **Die wahre Frequenz ist die der komplexen Komponente.** Das analytische Signal des reellen Cosinus weicht bei der Sinus-FM im ausgewerteten Fenster um bis zu 0,0013 davon ab (am Rand bis zu 0,003), das ist ein Teil des gemessenen Fehlers.
- **Rauschabstand gegen die Linie, nicht gegen die Gesamtleistung;** das Rauschen wird auf das reelle Signal addiert, bevor das analytische Signal entsteht. Die Rauschschwelle (3 Effektivwerte) ist eine Wahl, keine Optimierung.
- **Die Glättung σ_t = 6 der SPWVD gilt für diesen Abstand der Töne** (Periode 12,5 Abtastwerte), nicht allgemein.
- **Nur Gauß-Fenster** und ein σ je Lauf. Keine Umkehrung des Synchrosqueezing (Modenrekonstruktion), kein Reassignment oder Synchrosqueezing zweiter Ordnung, keine Wavelet-Variante (Synchrosqueezing der CWT).
- **Quellen** (Kodera, Gendrin und de Villedary 1976 und 1978; Auger und Flandrin 1995; Daubechies, Lu und Wu 2011; Ville 1948; Cohen 1966) wurden nicht gelesen; die Formeln sind per Handrechnung, Simulation und gegen geschlossene Ausdrücke geprüft.

## Verifikation

- **Kern:** STFT gegen die direkte Summe und gegen scipy (Fensterwerte, analytisches Signal), Verschiebungsinvarianz; Neuzuordnung: Impuls (t̂ gleich der Impulszeit), Ton, linearer Chirp (f̂ = f(t̂) bis 5·10⁻⁴), Energiebilanz des Bildes, Summe des Synchrosqueezing gleich der STFT-Summe; Wigner-Ville: Mini-Instanz von Hand, direkte Doppelsumme, Marginale, geschlossene Form des Kreuzterms und seiner Höhe; SPWVD: ohne Glättung gleich der Wigner-Ville, FFT-Glättung gegen direkte Faltung; Kennzahlen gegen handgebaute Bilder.
- **Experimente:** feste Werte (`tests/test_experiments_golden.py`), jede Zahl dieser README und der Texte in der App an die Auswertung gebunden (`tests/test_claims.py`), Verdikte an handgesetzten Zeilen.
- **Mutationstest:** `tools/mutate_demo.py` über alle Kernmodule (Ergebnis in `tests/mutation_state.json`).
- **Presets, Oberfläche:** Vollständigkeit und Grenzen, AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Permalink, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `rea_signals.py` | Testsignale, analytisches Signal, Rauschen |
| `rea_stft.py` | STFT mit Gauß-Fenster, Ableitung und Zeitgewichtung |
| `rea_reassign.py` | Reassignment, Synchrosqueezing, Rauschschwelle |
| `rea_wvd.py` | Wigner-Ville, SPWVD |
| `rea_measure.py` | Kennzahlen (wirksame Breite, Gipfel, Anteil auf den Linien, Kreuzterm), Kämme, Bilder der Verfahren |
| `rea_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Farben, Presets |
| `rea_evaluation.py` | Analyse, die fünf Experimente, Verdikte |
| `rea_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `rea_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `rea_pdf_export.py` | PDF-Export der Einstellung |
| `signal_generator.py` | gemeinsamer Generator (nur SplitMix64) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel (STFT, Neuzuordnung, Wigner-Ville, Kennzahlen, Signale, Auswertung), Experimente, Claims, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

Reassignment und Synchrosqueezing zweiter Ordnung; Umkehrung des Synchrosqueezing (Rekonstruktion einzelner Moden); Synchrosqueezing der CWT; weitere Cohen-Kerne (Choi-Williams, Born-Jordan); andere Fenster als Gauß; farbiges Rauschen;
Impulse und Amplitudenmodulation als Testsignale.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
