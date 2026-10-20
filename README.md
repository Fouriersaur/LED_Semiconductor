# LED semiconductor lab

Analysis scripts for the LED / diode lab sessions (October 2026). Run each script from
the repository root, e.g. `python3 IV_Curve_25C.py`; it reads from `data/` and writes its
plots to `plots/`.

## Scripts

| Script | What it does | Data | Plots |
|---|---|---|---|
| `IV_Curve_25C.py` | I–V fits (ideality factor, I0, series resistance) of the LEDs and Si diode | `data/LED`, `data/GaAs_LED`, `data/Si_diode` | `plots/IV/<csv name>/` |
| `two_diode_fit.py` | Two-diode fit of the GaAs LED temperature series | `data/GaAs_LED/T_series` | `plots/IV/` |
| `I0_vs_T.py` | Bandgap from I0(T) of the GaAs LED temperature series | `data/GaAs_LED/T_series` | `plots/IV/9_Oct_IV_GaAs_LED_I0_vs_T/` |
| `spectrum_bandgap.py` | Bandgap from the emission spectra, 15 Oct (84–102 K) | `data/LED_spectra_15Oct` | `plots/spectra/15_Oct_spectrum_LED/` |
| `spectrum_bandgap_16Oct.py` | Same, 16 Oct (77–269 K) | `data/LED_spectra_16Oct` | `plots/spectra/16_Oct_spectrum_LED/` |
| `forward_voltage_t0.py` | Forward voltage at the start of each current pulse, back extrapolated in √t | `data/LED_pulses` | `plots/pulses/` |

`IV_Curve_25C.py` also holds the shared plot style and constants that the other scripts
import.

## Layout

```
data/
  LED/, GaAs_LED/, Si_diode/   Keithley I–V sweeps
  LED_spectra_15Oct/           CCS200 spectra, real_<T>_pm<dT>.csv
  LED_spectra_16Oct/           CCS200 spectra, 161026_<T>_pm<dT>.csv, plus backgrounds
  LED_pulses/                  scope traces of the LED voltage under pulsed current
plots/
  IV/, spectra/, pulses/       outputs of the scripts above
```
