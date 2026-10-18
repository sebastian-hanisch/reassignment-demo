"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Bedingt gezeigte Regler (Abstand, Rauschabstand, SPWVD-Glättung, Leistungs- und Rauschschwelle) haben keinen eigenen Widget-Schlüssel: Ihr Wert steht unter `*_kept` und überlebt so die Läufe, in denen sie nicht erscheinen."""

import random
from dataclasses import dataclass

import streamlit as st

import rea_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "signal_select": SettingSpec("s", str, C.DEFAULT_SIGNAL, choices=C.SIGNALS),
    "sigma_slider": SettingSpec("sg", int, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "method_select": SettingSpec("m", str, C.DEFAULT_METHOD, choices=C.METHODS),
    "df_kept": SettingSpec("df", float, C.DEFAULT_DF, C.DF_MIN, C.DF_MAX),
    "noise_select": SettingSpec("rs", str, C.DEFAULT_NOISE, choices=C.NOISE_MODES),
    "snr_kept": SettingSpec("snr", float, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "sigma_t_kept": SettingSpec("st", int, C.DEFAULT_SIGMA_T, C.SIGMA_T_MIN, C.SIGMA_T_MAX),
    "sigma_lag_kept": SettingSpec("sl", int, C.DEFAULT_SIGMA_LAG, C.SIGMA_LAG_MIN, C.SIGMA_LAG_MAX),
    "floor_kept": SettingSpec("fl", int, C.DEFAULT_FLOOR, C.FLOOR_MIN, C.FLOOR_MAX),
    "gate_kept": SettingSpec("gt", str, C.DEFAULT_GATE, choices=C.GATE_MODES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_select", "signal": "signal_select", "sigma": "sigma_slider", "method": "method_select", "df": "df_kept", "noise": "noise_select", "snr": "snr_kept", "sigma_t": "sigma_t_kept",
               "sigma_lag": "sigma_lag_kept", "floor": "floor_kept", "gate": "gate_kept", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
