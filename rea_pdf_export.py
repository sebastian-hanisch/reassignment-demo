"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from rea_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "π": "pi", "≥": ">=",
            "λ": "lambda", "ν": "nu", "₂": "2", "₁": "1"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Reassignment und Synchrosqueezing: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    noise = f"Rauschen an, {de(s.snr, 0)} dB gegen die Linie" if s.noise_on else "ohne Rauschen"
    two = f", Abstand {de(s.df, 3)}" if s.two else ""
    spw = f", SPWVD sigma_t {s.sigma_t}, sigma_lag {s.sigma_lag}" if s.method == "SPWVD" else ""
    pdf.multi_cell(0, 5, _s(f"{s.signal}{two}, n = {s.n}, Fenster sigma = {s.sigma}, Verfahren {s.method}{spw}, Leistungsschwelle {s.floor} dB, {noise}, Rauschschwelle {s.gate}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [f"Wirksame Breite (Rasterpunkte 1/1024): {s.method} {de(a.width, 1)}, Spektrogramm {de(a.width_spec, 1)}.",
             f"Anteil auf den wahren Linien {de(100 * a.share, 0)} % (Spektrogramm {de(100 * a.share_spec, 0)} %)."]
    if s.signal == "Zwei Töne":
        lines.append(f"Gipfel im Zeitprofil: {a.peaks} (Spektrogramm {a.peaks_spec}), richtig sind 2.")
    if a.rmse is not None:
        lines.append(f"Frequenzfehler des Kamms (RMSE): {de(a.rmse, 5)}, Spektrogramm {de(a.rmse_spec, 5)}.")
    if a.cross is not None:
        lines.append(f"Kreuzterm (Mitte gegen Ton): {de(a.cross, 4)}, Spektrogramm {de(a.cross_spec, 4)}.")
    lines.append(f"Kleinster Wert gegen den größten: {de(a.neg, 3)}.")
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: reassignment-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
