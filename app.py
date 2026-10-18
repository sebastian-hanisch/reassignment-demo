"""Neu zuordnen statt verschmieren: Reassignment und Synchrosqueezing schärfen das Spektrogramm - und wo die Wigner-Ville-Verteilung scheitert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 14 der Reihe „Spektral- und Wavelet-Analyse“. Ein Spektrogramm mit Gauß-Fenster wird auf seinen Energieschwerpunkt neu zugeordnet (Reassignment), nur in der Frequenz verschoben (Synchrosqueezing)
und mit der Wigner-Ville-Verteilung und ihrer geglätteten Pseudo-Form (SPWVD) verglichen; fünf Experimente zu Schärfe, Kreuztermen, Rauschen, Trennung zweier Töne und Fensterbreite.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rea_constants as C
import rea_evaluation as ev
from rea_evaluation import Settings, analyse, de
from rea_pdf_export import build_pdf
from rea_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rea_visualization import (
    build_concentration,
    build_cross,
    build_images,
    build_noise,
    build_resolution,
    build_ridge,
    build_signal,
    build_slice,
    build_window,
)

st.set_page_config(page_title="Neu zuordnen statt verschmieren – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🎯 Neu zuordnen statt verschmieren")
st.markdown(
    """
Ein **Spektrogramm** verschmiert die Energie über die Breite des Fensters: Ein Ton, der nur einen Rasterpunkt breit ist, erscheint über Dutzende. **Reassignment** (Neuzuordnung) verschiebt jeden Wert
$(t, f)$ an den **Schwerpunkt** seiner Energie $(\\hat t, \\hat f)$, den man aus zwei weiteren Fensterversionen (Ableitung und Zeitgewicht) ausrechnet. **Synchrosqueezing** verschiebt nur in der Frequenz und
sammelt die komplexen Werte, damit die Darstellung umkehrbar bleibt. Die **Wigner-Ville-Verteilung** (WVD) braucht mathematisch kein Fenster und ist für einen Chirp schärfer als das Spektrogramm, aber nicht schärfer als die Neuzuordnung; sie erzeugt zwischen zwei Komponenten
**Kreuzterme** und negative Werte; die geglättete Pseudo-WVD (SPWVD) tauscht Schärfe gegen weniger Kreuzterme.

Dies ist die vierzehnte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf das **Spektrogramm** und das **Skalogramm** aufbauend. Alles ist **simuliert**: Die wahre Frequenz der Signale ist bekannt, die Verfahren
werden daran gemessen. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wird neu zugeordnet", expanded=True):
    st.markdown(
        """
1. **Spektrogramm:** Gauß-Fenster der Breite $\\sigma$, STFT $X(t, f)$; die Leistung $|X|^2$ ist über das Fenster verschmiert.
2. **Schwerpunkt:** Mit der Ableitung $g'$ und dem Zeitgewicht $t\\,g$ des Fensters ist $\\hat f = f - \\operatorname{Im}(X_{g'}/X_g)/2\\pi$ und $\\hat t = t + \\operatorname{Re}(X_{tg}/X_g)$ der Schwerpunkt der Energie um $(t, f)$.
3. **Neu zuordnen:** Die Leistung jedes Punktes wandert nach $(\\hat t, \\hat f)$; **Synchrosqueezing** nur nach $\\hat f$ (komplexer Wert). Bei einem linearen Chirp liegen die Punkte danach (bis auf die Ränder) in einem Band von wenigen Rasterpunkten um die Linie.
4. **Wigner-Ville:** $W(t, f) = \\sum_\\tau z[t{+}\\tau]\\,\\overline{z[t{-}\\tau]}\\,e^{-i4\\pi f\\tau}$ ohne Fenster in der Zeit; quadratisch im Signal, daher Kreuzterme. **SPWVD:** Gauß-Fenster in der Verzögerung und Glättung in der Zeit.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    kind = st.selectbox("Signal", list(C.SIGNALS), key="signal_select",
                        help="Ton: eine feste Frequenz. Chirp: lineare Frequenzrampe 0,05 bis 0,20. Sinus-FM: sinusförmig schwankende Frequenz. Zwei Töne und zwei parallele Chirps: zwei Komponenten im Abstand df.")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte.")
    if kind in C.SIGNAL_TWO:
        df = st.slider("Abstand der Komponenten df", C.DF_MIN, C.DF_MAX, float(st.session_state["df_kept"]), step=C.DF_STEP, format="%.3f",
                       help="Abstand in Zyklen je Abtastwert. Die Fensterbreite in der Frequenz ist 1/(2πσ) (bei σ = 12 etwa 0,013).")
        st.session_state["df_kept"] = float(df)
    else:
        df = float(st.session_state["df_kept"])
    noise = st.radio("Rauschen", list(C.NOISE_MODES), key="noise_select", horizontal=True, help="Weißes Gauß-Rauschen auf dem reellen Signal.")
    if noise == "an":
        snr = st.slider("Rauschabstand (dB)", C.SNR_MIN, C.SNR_MAX, float(st.session_state["snr_kept"]), step=C.SNR_STEP, help="Leistung einer Linie der Amplitude 1 gegen die des Rauschens.")
        st.session_state["snr_kept"] = float(snr)
    else:
        snr = float(st.session_state["snr_kept"])
    st.markdown("**Verfahren**")
    sigma = st.slider("Fensterbreite σ (Abtastwerte)", C.SIGMA_MIN, C.SIGMA_MAX, key="sigma_slider",
                      help="Standardabweichung des Gauß-Fensters. Breit: gute Frequenzauflösung, schlechte Zeitauflösung; schmal umgekehrt.")
    method = st.radio("Darstellung rechts", list(C.METHODS), key="method_select",
                      help="Spektrogramm: |STFT|². Neu zugeordnet: Leistung am Energieschwerpunkt. Synchrosqueezing: komplexe Werte nur in der Frequenz verschoben. Wigner-Ville: ohne Fenster, mit Kreuztermen. SPWVD: geglättet.")
    if method == "SPWVD":
        sigma_t = st.slider("SPWVD: Glättung in der Zeit σ_t", C.SIGMA_T_MIN, C.SIGMA_T_MAX, int(st.session_state["sigma_t_kept"]),
                            help="Gauß-Glättung der Verzögerungsprodukte in der Zeit; 0 heißt keine Glättung (nur das Verzögerungsfenster). Gegen Kreuzterme mit Periode 1/df hilft σ_t etwa die halbe Periode (nur für df = 0,08 gemessen).")
        sigma_lag = st.slider("SPWVD: Verzögerungsfenster σ_lag", C.SIGMA_LAG_MIN, C.SIGMA_LAG_MAX, int(st.session_state["sigma_lag_kept"]),
                              help="Gauß-Fenster über die Verzögerung τ: klein glättet in der Frequenz stark, groß nähert die WVD an.")
        st.session_state["sigma_t_kept"], st.session_state["sigma_lag_kept"] = int(sigma_t), int(sigma_lag)
    else:
        sigma_t, sigma_lag = int(st.session_state["sigma_t_kept"]), int(st.session_state["sigma_lag_kept"])
    if method in ("Neu zugeordnet", "Synchrosqueezing"):
        floor = st.slider("Leistungsschwelle (dB gegen die Spitze)", C.FLOOR_MIN, C.FLOOR_MAX, int(st.session_state["floor_kept"]), step=C.FLOOR_STEP,
                          help="Punkte mit weniger Leistung werden nicht neu zugeordnet; ohne Schwelle landen Rauschwerte mit unbestimmtem Schwerpunkt weit verstreut.")
        st.session_state["floor_kept"] = int(floor)
        if noise == "an":
            gate = st.radio("Rauschschwelle", list(C.GATE_MODES), index=list(C.GATE_MODES).index(st.session_state["gate_kept"]), horizontal=True,
                            help=f"Punkte mit Betrag unter {C.GATE_FACTOR:g} Effektivwerten der STFT des Rauschens werden nicht neu zugeordnet.")
            st.session_state["gate_kept"] = gate
        else:
            gate = str(st.session_state["gate_kept"])
    else:
        floor, gate = int(st.session_state["floor_kept"]), str(st.session_state["gate_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "signal_select": kind, "sigma_slider": int(sigma), "method_select": method, "df_kept": float(df), "noise_select": noise, "snr_kept": float(snr),
                   "sigma_t_kept": int(sigma_t), "sigma_lag_kept": int(sigma_lag), "floor_kept": int(floor), "gate_kept": gate, "seed_input": int(seed)})

settings = Settings(int(n), kind, int(sigma), method, float(df), noise, float(snr), int(sigma_t), int(sigma_lag), int(floor), gate, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
metrics = [("Wirksame Breite", de(a.width, 1), f"Spektrogramm {de(a.width_spec, 1)}",
            "Summe der Beträge gegen das Maximum je Zeitpunkt, gemittelt (Rasterpunkte 1/1024 der Frequenz): 1 heißt eine scharfe Linie. Misst Schärfe, nicht Richtigkeit."),
           ("Anteil auf den Linien", f"{de(100 * a.share, 0)} %", f"Spektrogramm {de(100 * a.share_spec, 0)} %",
            f"Anteil der Beträge innerhalb von ± {C.BAND} Rasterpunkten einer wahren Linie (zu jedem Zeitpunkt)."),
           ("Kleinster / größter Wert", de(a.neg, 3), None, "Null für Verteilungen ohne negative Werte; die Wigner-Ville-Verteilung hat negative Werte.")]
if a.rmse is not None:
    metrics.insert(1, ("Frequenzfehler", de(a.rmse, 5), f"Spektrogramm {de(a.rmse_spec, 5)}",
                       "Quadratischer Mittelwert der Abweichung des Gipfels von der wahren Frequenz (Zyklen je Abtastwert); das Raster gibt 0,00028 vor."))
if settings.signal == "Zwei Töne":
    metrics.insert(1, ("Gipfel im Zeitprofil", f"{a.peaks}", f"Spektrogramm {a.peaks_spec}", "Lokale Maxima der über die Zeit summierten Beträge (über 20 % des höchsten): zwei sind richtig."))
    metrics.append(("Kreuzterm / Ton", de(a.cross, 4), f"Spektrogramm {de(a.cross_spec, 4)}", "Betrag in der Mitte zwischen den Tönen gegen den am Ton; 1 heißt: so hoch wie der Ton selbst."))
for col, (label, value, delta, help_text) in zip(st.columns(len(metrics)), metrics):
    col.metric(label, value, delta=delta, delta_color="off", help=help_text)
st.plotly_chart(build_signal(a), width="stretch", key="signal")
st.plotly_chart(build_images(a), width="stretch", key="images")
st.caption("Links das Spektrogramm, rechts die gewählte Darstellung, beide auf gleichen Achsen; rot gestrichelt die wahre Frequenz. Die Wigner-Ville-Bilder sind vorzeichenbehaftet (rot positiv, blau negativ).")
st.plotly_chart(build_slice(a), width="stretch", key="slice")
st.caption("Schnitt durch beide Bilder in der Mitte der Aufzeichnung (Beträge in dB gegen den Gipfel): Breite der Linie und Nebenmaxima. Senkrecht gestrichelt die wahre Frequenz.")
ridge_fig = build_ridge(a)
if ridge_fig is not None:
    st.plotly_chart(ridge_fig, width="stretch", key="ridge")
    st.caption("Gipfel je Zeitpunkt (Kamm): Bei der Neuzuordnung ist es die neu zugeordnete Frequenz am Gipfel der STFT; an den Rändern der Aufzeichnung ist das Fenster abgeschnitten.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="reassignment.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was gewinnt man, wenn man das Spektrogramm neu zuordnet?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme, unabhängig von den Reglern. n = {C.EXP_N}, Gauß-Fenster σ = {C.EXP_SIGMA:g} (Breite in der Frequenz {de(ev.window_width(C.EXP_SIGMA), 4)}), Frequenzraster 1/{C.NFFT}, "
           "wirksame Breite in Rasterpunkten, Frequenzfehler in Zyklen je Abtastwert.")

st.markdown("#### Schärfe: wirksame Breite")
conc = _experiment("concentration")
st.caption("Ohne Rauschen: Summe der Beträge gegen das Maximum je Zeitpunkt, Zeiten 100 bis 411; 1 ist eine scharfe Linie. Die SPWVD hat σ_t = 6 und σ_lag = 20.")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], **{f"Spektrogramm σ = {g:g}": de(v, 1) for g, v in r["spec"].items()}, "Neu zugeordnet": de(r["reassigned"], 1), "Synchrosqueezing": de(r["squeezed"], 1),
                            "Wigner-Ville": de(r["wvd"], 1), "SPWVD": de(r["spwvd"], 1)} for r in conc]), hide_index=True, width="stretch")
st.plotly_chart(build_concentration(conc), width="stretch", key="exp_concentration")
kind_v, text = ev.verdict_concentration(conc)
getattr(st, kind_v)(text)

st.markdown("#### Kreuzterme der Wigner-Ville-Verteilung")
cross = _experiment("cross")
st.caption(f"Zwei gleich starke Töne bei {de(cross['f1'], 2)} und {de(cross['f2'], 2)}: Betrag in der Mitte zwischen ihnen gegen den am Ton (Mittel über die Zeiten 100 bis 411); darunter die SPWVD mit verschiedener Glättung.")
st.dataframe(pd.DataFrame([{"Verfahren": m["method"], "Kreuzterm / Ton": de(m["ratio"], 6), "wirksame Breite": de(m["width"], 1)} for m in cross["methods"]]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"σ_t": de(x["sigma_t"], 0), "σ_lag": de(x["sigma_lag"], 0), "Kreuzterm / Ton": de(x["ratio"], 4), "wirksame Breite": de(x["width"], 1)} for x in cross["smooth"]]), hide_index=True, width="stretch")
st.plotly_chart(build_cross(cross), width="stretch", key="exp_cross")
kind_v, text = ev.verdict_cross(cross)
getattr(st, kind_v)(text)

st.markdown("#### Frequenzschätzung im Rauschen")
noise_rows = _experiment("noise")
st.caption(f"Linearer Chirp 0,05 bis 0,20, Gipfel jedes Verfahrens als Frequenzschätzung, {noise_rows[0]['runs']} Läufe je Rauschabstand, Zeiten 60 bis 451; Rauschabstand gegen eine Linie der Amplitude 1. "
           f"Das Raster begrenzt den Fehler auf etwa {de(1 / (12 ** 0.5 * C.NFFT), 5)}. „mit Schwelle“: Synchrosqueezing ohne Punkte unter {C.GATE_FACTOR:g} Effektivwerten der STFT des Rauschens.")
st.dataframe(pd.DataFrame([{"Rauschabstand": f"{r['snr']} dB", **{k: de(r[k], 5) for k in ev.NOISE_METHODS}} for r in noise_rows]), hide_index=True, width="stretch")
st.plotly_chart(build_noise(noise_rows), width="stretch", key="exp_noise")
kind_v, text = ev.verdict_noise(noise_rows)
getattr(st, kind_v)(text)

st.markdown("#### Zwei Töne trennen")
res_rows = _experiment("resolution")
st.caption(f"Zwei gleich starke Töne ab {de(C.EXP_RES_F1, 2)} im Abstand df; Gipfel des über die Zeit summierten Betrags (über 20 % des höchsten) im Spektrogramm und im neu zugeordneten Bild (Leistung über −40 dB), "
           f"und Anteil der Beträge innerhalb von ± {C.BAND} Rasterpunkten der wahren Linien. Fensterbreiten: df geteilt durch 1/(2πσ).")
st.dataframe(pd.DataFrame([{"df": de(r["df"], 3), "Fensterbreiten": de(r["widths"], 1), "Gipfel Spektrogramm": r["peaks_spec"], "Gipfel neu zugeordnet": r["peaks_re"], "Anteil Spektrogramm": f"{de(100 * r['share_spec'], 0)} %",
                            "Anteil neu zugeordnet": f"{de(100 * r['share_re'], 0)} %"} for r in res_rows]), hide_index=True, width="stretch")
st.plotly_chart(build_resolution(res_rows), width="stretch", key="exp_resolution")
kind_v, text = ev.verdict_resolution(res_rows)
getattr(st, kind_v)(text)

st.markdown("#### Fensterbreite")
win = _experiment("window")
st.caption(f"Sinus-FM (Mitte 0,12, Hub 0,04, Modulation 0,01): Frequenzfehler des Gipfels in Abhängigkeit von σ, ohne Rauschen; rechts mit {win[0]['snr']:g} dB ({win[0]['runs']} Läufe). Spalte „neue Zeit“: "
           "die neu zugeordnete Frequenz gegen die wahre Frequenz zur neu zugeordneten Zeit.")
st.dataframe(pd.DataFrame([{"σ": r["sigma"], "Spektrogramm": de(r["clean"]["Spektrogramm"], 5), "Neu zugeordnet": de(r["clean"]["Neu zugeordnet"], 5), "Neu zugeordnet, neue Zeit": de(r["clean"]["Neu zugeordnet (zur neuen Zeit)"], 5),
                            "Synchrosqueezing": de(r["clean"]["Synchrosqueezing"], 5), f"Spektrogramm ({r['snr']:g} dB)": de(r["noisy"]["Spektrogramm"], 5), f"Synchrosqueezing ({r['snr']:g} dB)": de(r["noisy"]["Synchrosqueezing"], 5)}
                           for r in win]), hide_index=True, width="stretch")
st.plotly_chart(build_window(win), width="stretch", key="exp_window")
kind_v, text = ev.verdict_window(win)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Komponente je Zeitpunkt** | Bei zwei Komponenten innerhalb einer Fensterbreite mischen sich die Schwerpunkte: Das neu zugeordnete Bild zeigt Punkte, die zu keinem Signal gehören. | größeres Fenster, Zerlegung in Moden |
| **Die Linie krümmt sich nicht im Fenster** | Der Schwerpunkt-Ansatz ist für lineare Chirps exakt; bei Sinus-FM bleibt eine Restbreite, und der Fehler wächst mit der Fensterbreite. | Reassignment und Synchrosqueezing zweiter Ordnung |
| **Das Rauschen ist klein gegen die Linie** | Der Schwerpunkt aus Quotienten kleiner Werte ist unbestimmt: Synchrosqueezing verschlechtert sich ab etwa 6 dB und ist bei 0 dB etwa 30-mal schlechter als das Spektrogramm; bei −5 dB liegen alle Verfahren außer der SPWVD bei 0,06 bis 0,14. | Rauschschwelle, robuste Schätzung |
| **Neu zuordnen vergrößert die Auflösung** | Nein (gleich starke Töne, σ = 12): Zwei Töne unter etwa einer Fensterbreite bleiben einer, knapp darüber entstehen zusätzliche Gipfel. Die Auflösung setzt vermutlich das Fenster (nur eine Fensterbreite gemessen). | längeres Fenster, parametrische Verfahren |
| **Die Wigner-Ville-Verteilung ist das Schärfste** | Für einen einzelnen Chirp ist sie schärfer als das Spektrogramm, aber nicht schärfer als die Neuzuordnung; bei zwei Komponenten erzeugt sie Kreuzterme von der Größe der Komponenten und negative Werte. | SPWVD, Choi-Williams-Kern (nicht gebaut) |
| **Das Signal ist reell und analytisch gemacht** | Die Verteilungen gelten für das analytische Signal; die Frequenzen bleiben unter 0,25 (konservative Wahl), die Wigner-Ville-Verteilung des analytischen Signals faltet in dieser Formulierung erst am Rand 0 / 0,5. | höhere Abtastung |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**STFT.** Mit dem Gauß-Fenster $g(k) = e^{-k^2/(2\sigma^2)}$ ist $X_g(t, f) = \sum_k z[t{+}k]\,g(k)\,e^{-i2\pi f k}$ (Phasenbezug am Rahmenzentrum), das Spektrogramm $|X_g|^2$. Dasselbe mit $g'(k) = -\tfrac{k}{\sigma^2} g(k)$ und $(\mathcal T g)(k) = k\,g(k)$ gibt $X_{g'}$ und $X_{\mathcal T g}$.

**Reassignment.** Der Schwerpunkt der Energie um $(t, f)$ ist $\hat t = t + \operatorname{Re}\dfrac{X_{\mathcal T g}}{X_g}$ und $\hat f = f - \dfrac{1}{2\pi}\operatorname{Im}\dfrac{X_{g'}}{X_g}$. Das neu zugeordnete Spektrogramm sammelt
$|X_g|^2$ am nächsten Rasterpunkt von $(\hat t, \hat f)$. Für einen linearen Chirp ist $\hat f$ gleich der Momentanfrequenz zur Zeit $\hat t$ (exakt für das Gauß-Fenster).

**Synchrosqueezing.** $T(t, f) = \sum_{f' :\, \hat f(t, f') \approx f} X_g(t, f')$: nur in der Frequenz, komplexe Werte; die Summe über die Frequenz bleibt die der STFT (bis auf Punkte unter der Schwelle).

**Wigner-Ville.** $W(t, f) = \sum_\tau z[t{+}\tau]\,\overline{z[t{-}\tau]}\,e^{-i4\pi f\tau}$. Für $z = z_1 + z_2$ ist $W = W_{11} + W_{22} + 2\operatorname{Re} W_{12}$; der Kreuzterm $W_{12}$ liegt bei der mittleren Frequenz und schwingt mit
$\cos(2\pi(f_2 - f_1)t + \varphi)$, seine Höhe ist bis zu das Doppelte der Tonhöhe.

**SPWVD.** $\mathrm{SPWVD}(t, f) = \sum_\tau h(\tau)\,(g_t * R_\tau)(t)\,e^{-i4\pi f\tau}$ mit $R_\tau(t) = z[t{+}\tau]\overline{z[t{-}\tau]}$, Gauß-Fenstern $h$ (σ_lag) und $g_t$ (σ_t): Cohen-Klasse mit getrennt wählbarer Glättung in Zeit und Frequenz.

**Wirksame Breite.** $w = \frac{1}{|T|}\sum_t \sum_f |P(t, f)| / \max_f |P(t, f)|$ (Rasterpunkte). **Rauschschwelle.** $3\cdot 2\sigma_n\sqrt{\sum_k g(k)^2}$, das Dreifache des Effektivwerts von $|X_g|$ bei analytischem weißem Rauschen (reelle Standardabweichung $\sigma_n$).

**Quellen.** Lehrbuchwissen und Aufsätze (Kodera, Gendrin und de Villedary 1976 und 1978; Auger und Flandrin 1995; Daubechies, Lu und Wu 2011; Ville 1948; Cohen 1966). Formeln hier per Handrechnung, Simulation und gegen
geschlossene Ausdrücke geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
