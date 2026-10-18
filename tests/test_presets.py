"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import rea_constants as C
import rea_evaluation as ev
import rea_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["signal"] in C.SIGNALS and p["method"] in C.METHODS and p["noise"] in C.NOISE_MODES and p["gate"] in C.GATE_MODES, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.DF_MIN <= p["df"] <= C.DF_MAX and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and C.SIGMA_T_MIN <= p["sigma_t"] <= C.SIGMA_T_MAX, name
        assert C.SIGMA_LAG_MIN <= p["sigma_lag"] <= C.SIGMA_LAG_MAX and C.FLOOR_MIN <= p["floor"] <= C.FLOOR_MAX and (p["floor"] - C.FLOOR_MIN) % C.FLOOR_STEP == 0 and 0 <= p["seed"] <= C.SEED_MAX, name
        assert isinstance(p["sigma"], int) and isinstance(p["sigma_t"], int) and isinstance(p["sigma_lag"], int) and isinstance(p["floor"], int) and isinstance(p["df"], float) and isinstance(p["snr"], float), name


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.signal in C.SIGNALS and s.method in C.METHODS and s.noise in C.NOISE_MODES and s.gate in C.GATE_MODES
    assert C.SIGMA_MIN <= s.sigma <= C.SIGMA_MAX and C.DF_MIN <= s.df <= C.DF_MAX and C.SNR_MIN <= s.snr <= C.SNR_MAX and C.FLOOR_MIN <= s.floor <= C.FLOOR_MAX and set(C.SIGNAL_TWO) <= set(C.SIGNALS)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "signal": s.signal, "sigma": s.sigma, "method": s.method, "df": s.df, "noise": s.noise, "snr": s.snr, "sigma_t": s.sigma_t, "sigma_lag": s.sigma_lag, "floor": s.floor, "gate": s.gate,
                "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_not_password_like():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 70
    assert not {"pw", "pass", "password", "pwd", "token", "key"} & set(params)


def test_bounds_and_the_constant_relations():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("snr_kept") == (C.SNR_MIN, C.SNR_MAX) and P.bounds("sigma_slider") == (C.SIGMA_MIN, C.SIGMA_MAX) and P.bounds("n_select") == (None, None)
    assert C.DEFAULT_N in C.N_CHOICES and C.DEFAULT_METHOD in C.METHODS and C.DEFAULT_SIGNAL in C.SIGNALS and all(n & (n - 1) == 0 for n in C.N_CHOICES)
    assert 2 * -(-4 * C.SIGMA_MAX // 1) + 1 <= C.NFFT          # das längste Fenster passt in die FFT
    assert C.TWO_F1 + C.DF_MAX < 0.25 and C.PAR_F1 + C.DF_MAX < 0.25 and int(C.F_SHOW * C.NFFT) == 307
