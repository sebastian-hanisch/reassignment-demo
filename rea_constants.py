"""Konstanten: Regler-Grenzen, Signale, feste Annahmen, Experimente, Farben und Presets (Kern: rea_signals.py, rea_stft.py, rea_reassign.py, rea_wvd.py, rea_measure.py)."""

N_CHOICES = (256, 512, 1024)
DEFAULT_N = 512
SIGNALS = ("Ton", "Chirp", "Sinus-FM", "Zwei Töne", "Zwei parallele Chirps")
DEFAULT_SIGNAL = "Chirp"
SIGNAL_TWO = ("Zwei Töne", "Zwei parallele Chirps")                            # Signale mit zwei Komponenten (Abstand df, kein einzelner Kamm)
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 4, 32, 12                                # Standardabweichung des Gauß-Fensters in Abtastwerten
METHODS = ("Spektrogramm", "Neu zugeordnet", "Synchrosqueezing", "Wigner-Ville", "SPWVD")
DEFAULT_METHOD = "Neu zugeordnet"
DF_MIN, DF_MAX, DF_STEP, DEFAULT_DF = 0.005, 0.12, 0.005, 0.08                 # Abstand der beiden Komponenten (Zyklen je Abtastwert)
NOISE_MODES = ("aus", "an")
DEFAULT_NOISE = "aus"
SNR_MIN, SNR_MAX, SNR_STEP, DEFAULT_SNR = -5.0, 40.0, 1.0, 20.0                # dB, gegen eine Linie der Amplitude 1
SIGMA_T_MIN, SIGMA_T_MAX, DEFAULT_SIGMA_T = 0, 20, 6                           # SPWVD: Glättung in der Zeit (Abtastwerte)
SIGMA_LAG_MIN, SIGMA_LAG_MAX, DEFAULT_SIGMA_LAG = 5, 60, 20                    # SPWVD: Fenster in der Verzögerung (Abtastwerte)
FLOOR_MIN, FLOOR_MAX, FLOOR_STEP, DEFAULT_FLOOR = -80, -20, 5, -60             # Leistungsschwelle gegen die Spitze (dB)
GATE_MODES = ("aus", "an")
DEFAULT_GATE = "aus"
GATE_FACTOR = 3.0                                                              # Rauschschwelle: so viele Effektivwerte von |STFT| des Rauschens
SEED_MAX, DEFAULT_SEED = 999999, 41

NFFT = 1024                                                                    # Frequenzraster 1/1024 (Zyklen je Abtastwert)
F_SHOW = 0.3                                                                   # obere Frequenz der Bilder (die WVD ist nur unter 0,25 frei von Aliasing)
DB_FLOOR = -60.0
BAND = 3                                                                       # Halbbreite (Rasterpunkte) des Bandes um eine wahre Linie
OK_SHARE = 0.8                                                                 # Anteil auf den wahren Linien, ab dem das Verdikt „gut“ lautet
PEAK_LEVEL = 0.2                                                               # Gipfel zählen ab diesem Anteil des höchsten Gipfels

# Signale (feste Parameter)
TONE_F = 0.137
CHIRP_F0, CHIRP_F1 = 0.05, 0.20
SINFM_FC, SINFM_DF, SINFM_FM = 0.12, 0.04, 0.01
TWO_F1 = 0.10
PAR_F0, PAR_F1 = 0.04, 0.10

# Experimente (feste Ströme)
EXP_SEED = 1700
EXP_N, EXP_SIGMA = 512, 12.0
EXP_WIDTH_SIGMAS = (6.0, 12.0, 24.0)
EXP_SPWVD = ((0.0, 255.0), (0.0, 40.0), (0.0, 20.0), (6.0, 20.0), (12.0, 20.0), (12.0, 10.0))   # (sigma_t, sigma_lag) für die Kreuzterme
EXP_CROSS_TONES = (0.10, 0.18)
EXP_NOISE_SNRS, EXP_NOISE_RUNS = (60, 30, 20, 10, 0, -5), 20
EXP_RES_DFS = (0.01, 0.015, 0.02, 0.03, 0.05)
EXP_RES_F1 = 0.12
EXP_WINDOW_SIGMAS, EXP_WINDOW_RUNS, EXP_WINDOW_SNR = (4, 6, 8, 12, 16, 24, 32), 20, 10

COLOR_TRUE = "#B03A2E"
COLOR_EST = "#4C9F70"
COLOR_NOISY = "rgba(120,120,120,0.7)"
COLOR_AUX = "#E8833A"
COLOR_BLUE = "rgba(76,120,168,0.95)"

PRESETS = {
    "Chirp: Neu zuordnen schärft auf einen Rasterpunkt": {"n": 512, "signal": "Chirp", "sigma": 12, "method": "Neu zugeordnet", "df": 0.08, "noise": "aus", "snr": 20.0, "sigma_t": 6, "sigma_lag": 20, "floor": -60,
                                                         "gate": "aus", "seed": DEFAULT_SEED},
    "Sinus-FM: Synchrosqueezing und Neu zuordnen im Vergleich": {"n": 512, "signal": "Sinus-FM", "sigma": 12, "method": "Synchrosqueezing", "df": 0.08, "noise": "aus", "snr": 20.0, "sigma_t": 6, "sigma_lag": 20,
                                                                 "floor": -60, "gate": "aus", "seed": DEFAULT_SEED},
    "Zwei Töne: der Kreuzterm der Wigner-Ville-Verteilung": {"n": 512, "signal": "Zwei Töne", "sigma": 12, "method": "Wigner-Ville", "df": 0.08, "noise": "aus", "snr": 20.0, "sigma_t": 6, "sigma_lag": 20,
                                                            "floor": -60, "gate": "aus", "seed": DEFAULT_SEED},
    "Zwei Töne nahe beieinander: Neu zuordnen löst nicht auf": {"n": 512, "signal": "Zwei Töne", "sigma": 12, "method": "Neu zugeordnet", "df": 0.02, "noise": "aus", "snr": 20.0, "sigma_t": 6, "sigma_lag": 20,
                                                               "floor": -60, "gate": "aus", "seed": DEFAULT_SEED},
    "Rauschen: Synchrosqueezing bricht ein": {"n": 512, "signal": "Chirp", "sigma": 12, "method": "Synchrosqueezing", "df": 0.08, "noise": "an", "snr": 0.0, "sigma_t": 6, "sigma_lag": 20, "floor": -60,
                                              "gate": "aus", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Chirp: Neu zuordnen schärft auf einen Rasterpunkt": "Linearer Chirp 0,05 bis 0,20, Fenster σ = 12: Das Spektrogramm verschmiert die Linie über etwa 25 Rasterpunkte, die neu zugeordnete Darstellung zieht sie auf etwa einen zusammen.",
    "Sinus-FM: Synchrosqueezing und Neu zuordnen im Vergleich": "Sinusförmig modulierte Frequenz: Neu zuordnen schärft stark, Synchrosqueezing (nur in der Frequenz verschoben) weniger; an den Umkehrpunkten der Linie bleibt ein Rest an Breite.",
    "Zwei Töne: der Kreuzterm der Wigner-Ville-Verteilung": "Zwei gleich starke Töne bei 0,10 und 0,18: Die Wigner-Ville-Verteilung zeigt zwischen ihnen einen Kreuzterm von der Größe der Töne selbst, mit negativen Werten; das Spektrogramm hat dort fast nichts.",
    "Zwei Töne nahe beieinander: Neu zuordnen löst nicht auf": "Töne im Abstand 0,02 (1,5 Fensterbreiten): Das Spektrogramm trennt sie gerade noch, die neu zugeordnete Darstellung zeigt zusätzliche Gipfel, die es nicht gibt (Artefakte).",
    "Rauschen: Synchrosqueezing bricht ein": "Chirp bei 0 dB: Der Gipfel des Synchrosqueezings springt weit von der Linie weg (Fehler 0,08 gegen 0,0015 im Spektrogramm); schalten Sie die Rauschschwelle ein, um den Ausreißer zu entfernen. Im Mittel über 20 Läufe bleibt das Verfahren bei 0 dB auch mit Schwelle schlechter als das Spektrogramm (Experiment unten).",
}
