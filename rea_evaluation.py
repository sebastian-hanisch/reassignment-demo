"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import rea_constants as C
import rea_measure as M
import rea_reassign as R
import rea_signals as SG
import rea_stft as T
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; „–“ für nicht definierte Werte."""
    if x != x:
        return "–"
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    signal: str = C.DEFAULT_SIGNAL
    sigma: int = C.DEFAULT_SIGMA
    method: str = C.DEFAULT_METHOD
    df: float = C.DEFAULT_DF
    noise: str = C.DEFAULT_NOISE
    snr: float = C.DEFAULT_SNR
    sigma_t: int = C.DEFAULT_SIGMA_T
    sigma_lag: int = C.DEFAULT_SIGMA_LAG
    floor: int = C.DEFAULT_FLOOR
    gate: str = C.DEFAULT_GATE
    seed: int = C.DEFAULT_SEED

    @property
    def noise_on(self) -> bool:
        return self.noise == "an"

    @property
    def gate_on(self) -> bool:
        return self.gate == "an" and self.noise_on and self.method in ("Neu zugeordnet", "Synchrosqueezing")

    @property
    def two(self) -> bool:
        return self.signal in C.SIGNAL_TWO


def window_width(sigma: float) -> float:
    """Breite des Gauß-Fensters in der Frequenz (Standardabweichung des Betrags): 1 / (2 pi sigma) in Zyklen je Abtastwert."""
    return 1.0 / (2.0 * math.pi * sigma)


def build_measurement(s: Settings):
    """(Testsignal, gemessene reelle Reihe, analytisches Signal, Rauschstandardabweichung oder 0)."""
    sig = SG.make_signal(s.signal, s.n, s.df)
    sd = SG.noise_sigma(s.snr) if s.noise_on else 0.0
    x = sig.x + sd * G.SplitMix64(s.seed).normal(s.n) if s.noise_on else sig.x
    return sig, x, SG.analytic(x), sd


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    signal: SG.TestSignal
    x: np.ndarray
    img: np.ndarray                 # Bild des Verfahrens (n, 512)
    img_spec: np.ndarray            # Spektrogramm (n, 512)
    ridge: np.ndarray               # Kamm des Verfahrens
    ridge_spec: np.ndarray
    lo: int
    hi: int
    width: float                    # wirksame Breite des Verfahrens (Rasterpunkte)
    width_spec: float
    rmse: float | None              # Frequenzfehler des Kamms (nur bei einer Komponente)
    rmse_spec: float | None
    peaks: int                      # Gipfel im Zeitprofil
    peaks_spec: int
    share: float                    # Anteil auf den wahren Linien
    share_spec: float
    cross: float | None             # Verhältnis Mitte / Ton (nur zwei Töne)
    cross_spec: float | None
    neg: float                      # kleinster Wert gegen den größten
    gate_value: float               # verwendete Betragsschwelle (0 = keine)


def analyse(s: Settings) -> Analysis:
    sig, x, z, sd = build_measurement(s)
    lo, hi = M.inner(s.n)
    gate_value = R.noise_floor(float(s.sigma), sd) if s.gate_on else 0.0
    img, ridge = M.method_image(z, s.method, float(s.sigma), float(s.sigma_t), float(s.sigma_lag), float(s.floor), gate_value)
    img_spec, ridge_spec = M.method_image(z, "Spektrogramm", float(s.sigma))
    rlo, rhi = M.ridge_window(s.n)
    single = not s.two
    two_tones = s.signal == "Zwei Töne"
    f1, f2 = C.TWO_F1, C.TWO_F1 + s.df
    return Analysis(
        s, sig, x, img, img_spec, ridge, ridge_spec, lo, hi,
        M.effective_width(img, lo, hi), M.effective_width(img_spec, lo, hi),
        M.rmse(ridge, sig.tracks[0], rlo, rhi) if single else None, M.rmse(ridge_spec, sig.tracks[0], rlo, rhi) if single else None,
        M.count_peaks(M.time_profile(img, lo, hi)), M.count_peaks(M.time_profile(img_spec, lo, hi)),
        M.band_share(img, sig.tracks, lo, hi), M.band_share(img_spec, sig.tracks, lo, hi),
        M.cross_ratio(img, f1, f2, lo, hi) if two_tones else None, M.cross_ratio(img_spec, f1, f2, lo, hi) if two_tones else None,
        M.negative_depth(img, lo, hi), gate_value)


# --- Experimente (feste Ströme; Standardwerte aus rea_constants) ------------------------------------------------------------------------------


def _signal_z(kind: str, n: int = C.EXP_N, df: float = C.DEFAULT_DF):
    sig = SG.make_signal(kind, n, df)
    return sig, SG.analytic(sig.x)


def sweep_concentration(n: int = C.EXP_N, sigmas=C.EXP_WIDTH_SIGMAS, sigma: float = C.EXP_SIGMA) -> list[dict]:
    """E1: wirksame Breite (Rasterpunkte) je Verfahren für Chirp, Sinus-FM und Ton, ohne Rauschen."""
    lo, hi = M.inner(n)
    rows = []
    for kind in ("Chirp", "Sinus-FM", "Ton"):
        _sig, z = _signal_z(kind, n)
        row = {"signal": kind, "spec": {float(g): M.effective_width(T.spectrogram(z, g)[:, :M.H], lo, hi) for g in sigmas}}
        for key, method in (("reassigned", "Neu zugeordnet"), ("squeezed", "Synchrosqueezing"), ("wvd", "Wigner-Ville"), ("spwvd", "SPWVD")):
            img, _ = M.method_image(z, method, sigma)
            row[key] = M.effective_width(img, lo, hi)
        rows.append(row)
    return rows


def sweep_cross(n: int = C.EXP_N, sigma: float = C.EXP_SIGMA, tones=C.EXP_CROSS_TONES, smoothings=C.EXP_SPWVD) -> dict:
    """E2: Kreuzterm zwischen zwei gleich starken Tönen (Betrag in der Mitte gegen den am Ton), je Verfahren, und die SPWVD mit verschiedener Glättung."""
    f1, f2 = tones
    lo, hi = M.inner(n)
    z = SG.analytic(SG.two_tones(n, f1, f2).x)
    methods = []
    for method in ("Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Wigner-Ville"):
        img, _ = M.method_image(z, method, sigma)
        methods.append({"method": method, "ratio": M.cross_ratio(img, f1, f2, lo, hi), "width": M.effective_width(img, lo, hi)})
    wimg, _ = M.method_image(z, "Wigner-Ville", sigma)
    smooth = []
    for st, sl in smoothings:
        img, _ = M.method_image(z, "SPWVD", sigma, st, sl)
        smooth.append({"sigma_t": st, "sigma_lag": sl, "ratio": M.cross_ratio(img, f1, f2, lo, hi), "width": M.effective_width(img, lo, hi)})
    return {"f1": f1, "f2": f2, "methods": methods, "smooth": smooth, "negative": M.cross_negative(wimg, f1, f2, lo, hi), "period": 1.0 / (f2 - f1)}


NOISE_METHODS = ("Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Synchrosqueezing mit Schwelle", "Wigner-Ville", "SPWVD")


def sweep_noise(n: int = C.EXP_N, sigma: float = C.EXP_SIGMA, snrs=C.EXP_NOISE_SNRS, runs: int = C.EXP_NOISE_RUNS, seed: int = C.EXP_SEED) -> list[dict]:
    """E3: Frequenzschätzung (RMSE gegen die wahre Frequenz) des linearen Chirps im Rauschen, je Verfahren; Gipfel als Schätzung."""
    sig = SG.make_signal("Chirp", n)
    truth = sig.tracks[0]
    rows = []
    for snr in snrs:
        rng = G.SplitMix64(seed + int(snr))
        sd = SG.noise_sigma(snr)
        gate = R.noise_floor(sigma, sd)
        acc = {m: [] for m in NOISE_METHODS}
        for _ in range(runs):
            z = SG.analytic(sig.x + sd * rng.normal(n))
            err = M.ridge_errors(M.peak_estimates(z, sigma, gate), truth, n)
            for m in NOISE_METHODS:
                acc[m].append(err[m])
        rows.append({"snr": snr, "runs": runs, **{m: float(np.mean(acc[m])) for m in NOISE_METHODS}})
    return rows


def sweep_resolution(n: int = C.EXP_N, sigma: float = C.EXP_SIGMA, dfs=C.EXP_RES_DFS, f1: float = C.EXP_RES_F1) -> list[dict]:
    """E4: Trennung zweier Töne im Abstand df: Gipfel des Zeitprofils im Spektrogramm und im neu zugeordneten Spektrogramm (Leistung über −40 dB), Anteil auf den wahren Linien."""
    lo, hi = M.inner(n)
    rows = []
    for df in dfs:
        sig = SG.two_tones(n, f1, f1 + df)
        z = SG.analytic(sig.x)
        X, f_hat, t_hat, mask = R.reassign_fields(z, sigma, C.NFFT, -40.0)
        spec, re = (np.abs(X) ** 2)[:, :M.H], R.reassigned_from(X, t_hat, f_hat, mask)[:, :M.H]
        rows.append({"df": df, "widths": df * 2.0 * math.pi * sigma, "peaks_spec": M.count_peaks(M.time_profile(spec, lo, hi)), "peaks_re": M.count_peaks(M.time_profile(re, lo, hi)),
                     "share_spec": M.band_share(spec, sig.tracks, lo, hi), "share_re": M.band_share(re, sig.tracks, lo, hi)})
    return rows


def sweep_window(n: int = C.EXP_N, sigmas=C.EXP_WINDOW_SIGMAS, runs: int = C.EXP_WINDOW_RUNS, snr: float = C.EXP_WINDOW_SNR, seed: int = C.EXP_SEED) -> list[dict]:
    """E5: Frequenzfehler der Sinus-FM in Abhängigkeit von der Fensterbreite: Spektrogramm-Gipfel, neu zugeordnete Frequenz (gegen die Frequenz zur Rahmenzeit und zur neuen Zeit), Synchrosqueezing; ohne Rauschen und mit snr dB."""
    sig = SG.make_signal("Sinus-FM", n)
    truth = sig.tracks[0]
    z0 = SG.analytic(sig.x)
    keys = ("Spektrogramm", "Neu zugeordnet", "Neu zugeordnet (zur neuen Zeit)", "Synchrosqueezing")
    rows = []
    for g in sigmas:
        clean = M.ridge_errors(M.peak_estimates(z0, float(g), with_wvd=False), truth, n)
        rng = G.SplitMix64(seed + 100 + int(g))
        sd = SG.noise_sigma(snr)
        acc = {k: [] for k in keys}
        for _ in range(runs):
            err = M.ridge_errors(M.peak_estimates(SG.analytic(sig.x + sd * rng.normal(n)), float(g), with_wvd=False), truth, n)
            for k in keys:
                acc[k].append(err[k])
        rows.append({"sigma": g, "runs": runs, "snr": snr, "clean": {k: clean[k] for k in keys}, "noisy": {k: float(np.mean(acc[k])) for k in keys}})
    return rows


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    base = f"{s.method} bei σ = {s.sigma}: wirksame Breite {de(a.width, 1)} Rasterpunkte, Spektrogramm {de(a.width_spec, 1)}."
    if s.method == "Spektrogramm":
        if a.rmse is not None:
            extra = f" Frequenzfehler des Kamms {de(a.rmse, 5)}."
        elif s.signal == "Zwei Töne":
            extra = f" Gipfel im Zeitprofil: {a.peaks} (wahr: 2)."
        else:
            extra = f" Anteil auf den wahren Linien {de(100 * a.share, 0)} %."
        return "info", base + extra + " Die Breite folgt dem Fenster (1/(2πσ) = " + de(window_width(s.sigma), 4) + " in der Frequenz); die Neuzuordnung ist der Vergleich."
    if s.two:
        line_share = f" Anteil auf den wahren Linien {de(100 * a.share, 0)} % (Spektrogramm {de(100 * a.share_spec, 0)} %)."
        if a.cross is not None and s.method in ("Wigner-Ville", "SPWVD") and a.cross > 0.1:
            return "warning", base + f" Zwischen den Tönen liegt ein Kreuzterm von {de(a.cross, 2)} der Tonhöhe (Spektrogramm {de(a.cross_spec, 4)})" + (", mit negativen Werten." if a.neg < -0.05 else ".") + line_share
        if s.signal == "Zwei Töne" and a.peaks != 2:
            return "warning", base + f" Das Zeitprofil zeigt {a.peaks} Gipfel statt 2: " + ("die Töne sind nicht getrennt." if a.peaks < 2 else "es sind Artefakte dabei, die es im Signal nicht gibt.") + line_share
        return ("success" if a.share >= C.OK_SHARE else "info"), base + (" Zwei Gipfel." if s.signal == "Zwei Töne" else "") + line_share
    err = f" Frequenzfehler des Kamms {de(a.rmse, 5)} (Spektrogramm {de(a.rmse_spec, 5)})."
    if a.rmse > 3 * a.rmse_spec:
        return "warning", base + err + " Der Gipfel liegt weit neben der Linie: Rauschwerte bestimmen ihn."
    if a.width < 0.5 * a.width_spec:
        return "success", base + err
    return "info", base + err


def verdict_concentration(rows: list[dict]) -> tuple[str, str]:
    sp = C.EXP_SIGMA
    by = {r["signal"]: r for r in rows}
    ch, fm = by["Chirp"], by["Sinus-FM"]
    ok = all(r["reassigned"] < r["spec"][sp] for r in rows)
    return ("success" if ok else "warning"), (f"Beim Chirp schrumpft die wirksame Breite von {de(ch['spec'][sp], 1)} (Spektrogramm, σ = {sp:g}) auf {de(ch['reassigned'], 1)} (Neu zuordnen) und {de(ch['squeezed'], 1)} (Synchrosqueezing); "
                                              f"Wigner-Ville {de(ch['wvd'], 1)}. Bei der Sinus-FM bleibt mehr übrig: {de(fm['spec'][sp], 1)} auf {de(fm['reassigned'], 1)} und {de(fm['squeezed'], 1)} (Wigner-Ville {de(fm['wvd'], 1)}), "
                                              "vermutlich, weil die Linie sich innerhalb des Fensters krümmt.")


def verdict_cross(res: dict) -> tuple[str, str]:
    by = {m["method"]: m for m in res["methods"]}
    st = next((x for x in res["smooth"] if x["sigma_t"] > 0 and x["ratio"] < 0.1), None)
    tail = (f" Eine Glättung in der Zeit (σ_t = {st['sigma_t']:g}, Verzögerungsfenster {st['sigma_lag']:g}) drückt ihn auf {de(st['ratio'], 3)}, die Töne werden dafür breiter ({de(st['width'], 1)} Rasterpunkte)." if st else "")
    return "warning", (f"Die Wigner-Ville-Verteilung zeigt zwischen den Tönen einen Kreuzterm von {de(by['Wigner-Ville']['ratio'], 2)} der Tonhöhe, mit Werten bis {de(res['negative'], 1)} der Tonhöhe; das Spektrogramm hat dort {de(by['Spektrogramm']['ratio'], 4)}. "
                       f"Der Kreuzterm schwingt in der Zeit mit der Periode 1/(f₂ − f₁) = {de(res['period'], 1)} Abtastwerten." + tail)


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    by = {r["snr"]: r for r in rows}
    hi, lo = by[max(by)], by[0]
    bad = lo["Synchrosqueezing"] > 5 * lo["Spektrogramm"]
    same = lo["Neu zugeordnet"] > 0.9 * lo["Spektrogramm"]
    gap = "bei 0 dB ist der Vorsprung " + ("weg" if same else "kleiner") + f" ({de(lo['Neu zugeordnet'], 4)} gegen {de(lo['Spektrogramm'], 4)})"
    return ("warning" if bad else "info"), (f"Bei hohem Rauschabstand ({max(by)} dB) ist die neu zugeordnete Frequenz am Gipfel genauer als der Gipfel des Spektrogramms ({de(hi['Neu zugeordnet'], 5)} gegen {de(hi['Spektrogramm'], 5)}, das Raster gibt "
                                            f"{de(1 / (math.sqrt(12) * C.NFFT), 5)} vor); {gap}. "
                                            f"Synchrosqueezing bricht bei 0 dB ein ({de(lo['Synchrosqueezing'], 4)}); mit einer Schwelle von {C.GATE_FACTOR:g} Effektivwerten des Rauschens in der STFT sind es {de(lo['Synchrosqueezing mit Schwelle'], 4)}.")


def verdict_resolution(rows: list[dict]) -> tuple[str, str]:
    first_spec = next((r for r in rows if r["peaks_spec"] >= 2), None)
    extra = [r for r in rows if r["peaks_re"] > 2]
    parts = []
    if first_spec:
        parts.append(f"Das Spektrogramm zeigt zwei Gipfel ab df = {de(first_spec['df'], 3)} ({de(first_spec['widths'], 1)} Fensterbreiten).")
    if extra:
        e = extra[0]
        parts.append(f"Das neu zugeordnete Bild zeigt bei df = {de(e['df'], 3)} {e['peaks_re']} Gipfel statt 2: Artefakte, kein Gewinn an Auflösung.")
    far = rows[-1]
    parts.append(f"Erst bei df = {de(far['df'], 3)} liegen {de(100 * far['share_re'], 0)} % der neu zugeordneten Leistung auf den wahren Linien (Spektrogramm {de(100 * far['share_spec'], 0)} %).")
    return "warning", " ".join(parts)


def verdict_window(rows: list[dict]) -> tuple[str, str]:
    first, last = rows[0], rows[-1]
    same = max(abs(r["clean"]["Neu zugeordnet"] - r["clean"]["Spektrogramm"]) / r["clean"]["Spektrogramm"] for r in rows)
    return "info", (f"Der Frequenzfehler des Gipfels wächst mit der Fensterbreite: {de(first['clean']['Spektrogramm'], 5)} bei σ = {first['sigma']:g}, {de(last['clean']['Spektrogramm'], 5)} bei σ = {last['sigma']:g}. "
                    f"Die neu zugeordnete Frequenz am Gipfel ändert daran höchstens {de(100 * same, 1)} %; erst gegen die Frequenz zur neu zugeordneten Zeit sinkt der Fehler bei σ = {last['sigma']:g} auf {de(last['clean']['Neu zugeordnet (zur neuen Zeit)'], 5)}.")
