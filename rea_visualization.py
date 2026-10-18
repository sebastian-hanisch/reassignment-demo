"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rea_constants as C
from rea_evaluation import Analysis, de

SIGNED_METHODS = ("Wigner-Ville", "SPWVD")


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def power_db(img: np.ndarray) -> np.ndarray:
    """Leistung in dB gegen das Maximum, nach unten bei DB_FLOOR abgeschnitten."""
    top = float(img.max())
    if top <= 0.0:
        return np.full(img.shape, C.DB_FLOOR)
    return np.maximum(10.0 * np.log10(np.maximum(img, 1e-300) / top), C.DB_FLOOR)


def display_image(img: np.ndarray, signed: bool) -> tuple[np.ndarray, dict]:
    """(z, Darstellung): nichtnegative Bilder in dB (Viridis), vorzeichenbehaftete (Wigner-Ville) linear auf das größte |Wert| normiert (blau negativ, rot positiv)."""
    cols = int(C.F_SHOW * C.NFFT)
    sub = img[:, :cols]
    if signed:
        top = float(np.abs(sub).max())
        z = sub / top if top > 0 else sub
        return z, dict(colorscale="RdBu_r", zmin=-1.0, zmax=1.0, colorbar=dict(title="W / max", thickness=12))
    return power_db(sub), dict(colorscale="Viridis", zmin=C.DB_FLOOR, zmax=0.0, colorbar=dict(title="dB", thickness=12))


def build_signal(a: Analysis) -> go.Figure:
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.x, mode="lines", line=dict(color=C.COLOR_BLUE, width=1), name="Messsignal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    if a.settings.noise_on:
        fig.add_trace(go.Scatter(x=t, y=a.signal.x, mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dot"), name="sauberes Signal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_vrect(x0=a.lo, x1=a.hi - 1, fillcolor=C.COLOR_AUX, opacity=0.10, line_width=0)
    fig.update_xaxes(title_text="Zeit (Abtastwerte); orange: Zeitfenster der Kennzahlen", range=[0, len(t) - 1])
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=220)


def build_images(a: Analysis) -> go.Figure:
    """Spektrogramm (links) und Bild des gewählten Verfahrens (rechts) auf gleichen Achsen; darüber die wahren Linien (rot gestrichelt) und der Kamm."""
    s = a.settings
    stride = max(1, len(a.x) // 512)
    t = np.arange(len(a.x))[::stride]
    freqs = np.arange(int(C.F_SHOW * C.NFFT)) / C.NFFT
    titles = ("Spektrogramm", s.method)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06, subplot_titles=titles)
    for col, (img, signed) in enumerate(((a.img_spec, False), (a.img, s.method in SIGNED_METHODS)), start=1):
        z, style = display_image(img, signed)
        style = dict(style, showscale=col == 2)
        fig.add_trace(go.Heatmap(x=t, y=freqs, z=z[::stride].T, hovertemplate="t = %{x:.0f}, f = %{y:.3f}: %{z:.2f}<extra></extra>", **style), row=1, col=col)
        for i, track in enumerate(a.signal.tracks):
            fig.add_trace(go.Scatter(x=t, y=np.asarray(track)[::stride], mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dash"), name="wahre Frequenz", showlegend=(col == 1 and i == 0),
                                     hoverinfo="skip"), row=1, col=col)
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(a.x) - 1])
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, C.F_SHOW], col=1)
    return _fixed(fig, height=430)


def build_slice(a: Analysis) -> go.Figure:
    """Frequenzschnitt in der Mitte der Aufzeichnung (Beträge in dB gegen den Gipfel): Spektrogramm gegen das gewählte Verfahren; senkrecht die wahre Frequenz."""
    t = len(a.x) // 2
    cols = int(C.F_SHOW * C.NFFT)
    freqs = np.arange(cols) / C.NFFT

    def db(v):
        v = np.abs(v[t, :cols])
        top = float(v.max())
        return np.full(cols, C.DB_FLOOR) if top <= 0 else np.maximum(10.0 * np.log10(np.maximum(v, 1e-300) / top), C.DB_FLOOR)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=db(a.img_spec), mode="lines", line=dict(color=C.COLOR_NOISY, width=1.6), name="Spektrogramm", hovertemplate="f = %{x:.3f}: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=freqs, y=db(a.img), mode="lines", line=dict(color=C.COLOR_EST, width=1.6), name=a.settings.method, hovertemplate="f = %{x:.3f}: %{y:.1f} dB<extra></extra>"))
    for track in a.signal.tracks:
        fig.add_vline(x=float(track[t]), line=dict(color=C.COLOR_TRUE, width=1, dash="dash"))
    fig.update_xaxes(title_text=f"Frequenz (Zyklen je Abtastwert), Schnitt bei t = {t}", range=[0, C.F_SHOW])
    fig.update_yaxes(title_text="Betrag (dB gegen den Gipfel)", range=[C.DB_FLOOR - 2, 2])
    return _fixed(fig, height=280)


def build_ridge(a: Analysis) -> go.Figure | None:
    """Kamm (Gipfel je Zeitpunkt) des Verfahrens und des Spektrogramms gegen die wahre Frequenz; nur bei einer Komponente."""
    if a.settings.two:
        return None
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.signal.tracks[0], mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), name="wahre Frequenz", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.ridge_spec, mode="lines", line=dict(color=C.COLOR_NOISY, width=1.2), name="Kamm des Spektrogramms", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.ridge, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name=f"Kamm: {a.settings.method}", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(t) - 1])
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)")
    return _fixed(fig, height=260)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_concentration(rows) -> go.Figure:
    sp = C.EXP_SIGMA
    fig = go.Figure()
    series = [(f"Spektrogramm σ = {g:g}", [r["spec"][g] for r in rows]) for g in C.EXP_WIDTH_SIGMAS]
    series += [("Neu zugeordnet", [r["reassigned"] for r in rows]), ("Synchrosqueezing", [r["squeezed"] for r in rows]), ("Wigner-Ville", [r["wvd"] for r in rows]), ("SPWVD", [r["spwvd"] for r in rows])]
    for label, ys in series:
        fig.add_trace(go.Bar(x=[r["signal"] for r in rows], y=ys, name=label, hovertemplate=label + ", %{x}: %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title_text="wirksame Breite (Rasterpunkte, log)", type="log", range=[-0.2, 2.0])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)


def build_cross(res) -> go.Figure:
    labels = [m["method"] for m in res["methods"]] + [f"SPWVD σ_t = {x['sigma_t']:g}, σ_lag = {x['sigma_lag']:g}" for x in res["smooth"]]
    values = [max(m["ratio"], 1e-9) for m in res["methods"]] + [max(x["ratio"], 1e-9) for x in res["smooth"]]
    colors = [C.COLOR_TRUE if v > 0.1 else C.COLOR_EST for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, hovertemplate="%{x}: %{y:.3g}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dot"), annotation_text="Höhe des Tons", annotation_position="top left")
    fig.update_yaxes(title_text="Kreuzterm / Ton (log; unter 1e-9 abgeschnitten)", type="log", range=[-9.5, 0.5])
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=360)


def build_noise(rows) -> go.Figure:
    fig = go.Figure()
    for key in ("Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Synchrosqueezing mit Schwelle", "Wigner-Ville", "SPWVD"):
        fig.add_trace(go.Scatter(x=[r["snr"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=key, hovertemplate=key + ", %{x} dB: %{y:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="Rauschabstand gegen die Linie (dB)", autorange="reversed")
    fig.update_yaxes(title_text="Frequenzfehler (RMSE, log)", type="log")
    return _fixed(fig, height=360)


def build_resolution(rows) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["df"] for r in rows], y=[r["share_spec"] for r in rows], mode="lines+markers", name="Spektrogramm", hovertemplate="df = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["df"] for r in rows], y=[r["share_re"] for r in rows], mode="lines+markers", name="Neu zugeordnet", hovertemplate="df = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Abstand der Töne (Zyklen je Abtastwert)")
    fig.update_yaxes(title_text="Anteil auf den wahren Linien", range=[0, 1])
    return _fixed(fig, height=320)


def build_window(rows) -> go.Figure:
    fig = go.Figure()
    for key, label in (("Spektrogramm", "Spektrogramm-Gipfel"), ("Neu zugeordnet", "neu zugeordnet (zur Rahmenzeit)"), ("Neu zugeordnet (zur neuen Zeit)", "neu zugeordnet (zur neuen Zeit)"), ("Synchrosqueezing", "Synchrosqueezing")):
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rows], y=[r["clean"][key] for r in rows], mode="lines+markers", name=label, hovertemplate=label + ", σ = %{x}: %{y:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="Fensterbreite σ (Abtastwerte)")
    fig.update_yaxes(title_text="Frequenzfehler (RMSE, Sinus-FM, ohne Rauschen)", rangemode="tozero")
    return _fixed(fig, height=340)
