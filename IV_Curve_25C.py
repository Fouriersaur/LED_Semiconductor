"""Plot and fit the I-V characteristics of LEDs and diodes from Keithley SMU buffer exports."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# CSVs are grouped by device under data/: data/LED, data/Si_diode, data/GaAs_LED
# (with the temperature series in data/GaAs_LED/T_series)
DATA_DIR = Path(__file__).with_name("data")
LED_DIR = DATA_DIR / "LED"
SI_DIR = DATA_DIR / "Si_diode"
GAAS_DIR = DATA_DIR / "GaAs_LED"

DATA_FILES = [
    # LED_DIR / "6_Oct_IV_25C.csv",  # current sweep, voltage measured
    LED_DIR / "8_Oct_IV_25C.csv",  # voltage sweep, current measured
    LED_DIR / "8_Oct_IV_25C_Version2.csv",  # voltage sweep, current measured
    GAAS_DIR / "8_Oct_IV_25C_GaAs_LED.csv",  # voltage sweep, current measured
    GAAS_DIR / "8_Oct_IV_25C_GaAs_LED_Version2.csv",  # voltage sweep, current measured
    SI_DIR / "9_Oct_IV_25C_Silicon.csv",  # voltage sweep, current measured
    GAAS_DIR / "9_Oct_IV_25C_GaAs_LED_Version3.csv",  # voltage sweep, current measured
    GAAS_DIR / "9_Oct_IV_25C_GaAs_LED_T25.csv",  # voltage sweep, current measured
    GAAS_DIR / "9_Oct_IV_45C_GaAs_LED_T45.csv",  # voltage sweep, current measured
]
# GaAs LED temperature series (9-10 Oct), temperature in K in the name: 296.5K = 296.5 K
GAAS_T_SERIES_K = (296.5, 305.7, 314.7, 324.1, 333.5, 342.2, 351.4, 360.3)
GAAS_T_SERIES_FILES = [GAAS_DIR / "T_series" / f"9_Oct_IV_GaAs_LED_{t:g}K.csv"
                       for t in GAAS_T_SERIES_K]
DATA_FILES += GAAS_T_SERIES_FILES
# Plots go to plots/<csv name>/, e.g. plots/8_Oct_IV_25C/8_Oct_IV_25C_fit.png
PLOTS_DIR = Path(__file__).with_name("plots")

T_KELVIN = 298.15  # 25 °C, the default measurement temperature
K_B_OVER_Q = 8.617333262e-5  # V/K

# Per-file analysis settings, keyed by CSV name:
#   offset_window_v  voltages (V) where the device passes no measurable current, so the
#                    measured current there is the SMU's zero offset; its median is
#                    subtracted from all currents before the fit and the local ideality
#                    plot. None: no offset correction.
#   fit_range_a      Eq. (4) fit window (A), where the local ideality factor is flat.
#   window_lows_a, window_highs_a
#                    alternative fit windows (A), used to estimate the systematic error
#                    from the choice of range.
#   device           name used in the plot titles (default "LED").
#   temperature_c    measurement temperature (°C, default 25), used for kT/q.
#   temperature_k    optional: measurement temperature (K), instead of temperature_c.
#   sweep_start_v    optional: voltage (V) the sweep started at; readings below it are
#                    bogus readbacks that the sweep-order filter cannot catch because
#                    they come first.
#   local_n_min_current_a
#                    optional: lowest current (A) in the local ideality plot, instead of
#                    LOCAL_N_MIN_CURRENT_A.
#   rs_fit_range_a, rs_window_lows_a, rs_window_highs_a
#                    optional: fit window and alternative windows (A) for eq. (4) with a
#                    series resistance, V = (n_id kT/q) ln(I/I0) + I Rs.
LED_SETTINGS = dict(
    offset_window_v=None,
    # 0.15 mA: on 6 Oct the first point (80 µA, straight after the 0 A step) is an outlier
    fit_range_a=(0.15e-3, 0.55e-3),
    window_lows_a=(0.0, 0.15e-3, 0.25e-3),
    window_highs_a=(0.45e-3, 0.55e-3, 0.65e-3),
)
SETTINGS = {
    "6_Oct_IV_25C.csv": LED_SETTINGS,
    # The LED passes < ~30 pA up to 1.05 V (from the fit), but the SMU reads ~-1.2 nA there
    "8_Oct_IV_25C.csv": {**LED_SETTINGS, "offset_window_v": (0.1, 1.05)},
    # Conducts from ~0.1 V with n_id = 1.00 at 1-100 µA. No offset correction needed:
    # it reads -0.02 nA at 0 V, against 2 nA at the first step (2 mV).
    "8_Oct_IV_25C_Version2.csv": dict(
        offset_window_v=None,
        fit_range_a=(2e-6, 5e-5),
        window_lows_a=(1e-6, 2e-6, 5e-6),
        window_highs_a=(2e-5, 5e-5, 1e-4),
    ),
    # Reads ~-2.0 nA up to 0.35 V on the 100 nA range (the 0 V point, on the 1 µA
    # range, reads -0.3 nA and is left out); the diode itself passes < ~0.05 nA there.
    # The local n_id slides from 1.75 (0.1 µA) to 1.48 (3 mA) with no true plateau;
    # 5-60 µA (0.85-0.95 V) is the flattest stretch.
    "8_Oct_IV_25C_GaAs_LED.csv": dict(
        offset_window_v=(0.01, 0.35),
        fit_range_a=(5e-6, 6e-5),
        window_lows_a=(2e-6, 5e-6, 1e-5),
        window_highs_a=(3e-5, 6e-5, 1e-4),
    ),
}
# Same LED, measured again. Every step at which the SMU autoranged down to 100 nA
# (78 of 245) reads back a bogus -1.1 to -1.3 V, +27-50 nA; the sweep-order filter
# in analyse() drops them.
SETTINGS["8_Oct_IV_25C_GaAs_LED_Version2.csv"] = SETTINGS["8_Oct_IV_25C_GaAs_LED.csv"]
# Silicon diode (1N4001). No offset correction: the diode conducts from 0 V, so there is
# no window to take it from, but fitting the 0-0.15 V data as two-diode current plus a
# constant gives an offset of -2 ± 3 pA (the 0 V point, -0.006 nA, was read on the
# 1 µA range, so on its own it says little about the 100 nA range). The local
# n_id is ~1.4-1.5 at nA-µA, then a flat ~1.86 at 1-15 mA (high injection) before
# series resistance takes over. Eq. (4) is fitted on the flat bottom of the low-current
# minimum, where diffusion contributes most: 0.36-0.39 V (3.1-7.3 µA), where the local
# n_id stays within 0.005 of its minimum (1.384 at 0.375 V). The alternative windows
# widen this to within 0.01 (0.355-0.395 V) and 0.02 (0.345-0.405 V) of the minimum.
# Eq. (4) + Rs is fitted over the whole high-injection region,
# 1-100 mA. The 14 readings taken on the 10 nA range have bogus voltages and are
# dropped by the sweep-order filter.
SETTINGS["9_Oct_IV_25C_Silicon.csv"] = dict(
    offset_window_v=None,
    fit_range_a=(3.1e-6, 7.31e-6),
    window_lows_a=(2.0e-6, 2.7e-6, 3.1e-6),
    window_highs_a=(7.31e-6, 8.4e-6, 1.12e-5),
    rs_fit_range_a=(1e-3, 1e-1),
    rs_window_lows_a=(1e-3, 5e-3, 2e-2),
    rs_window_highs_a=(5e-2, 7e-2, 1e-1),
    device="Si diode (1N4001)",
)
# GaAs LED again, sweep from 0.5 V. Its first two readings (0.415, 0.418 V) and two
# more later were taken on the 10 nA range and have bogus voltages. No offset
# correction: the sweep starts where the LED already passes ~1 nA, so there is no
# window to take it from. The local n_id is flat at ~2.0 over 0.68-0.94 V
# (0.13-20 µA), dips to 1.56 at 1.12 V (~1 mA), then rises with series resistance.
# Eq. (4) is fitted on the bottom of that dip, 1.105-1.125 V (0.74-1.22 mA), where the
# local n_id stays within 0.01 of its minimum (1.563 at 1.115 V); the alternative
# windows are within 0.005 and 0.02 of it. The dip is only where I Rs starts to take
# over: with Rs (~2.4 Ω) removed, the junction n_id keeps falling to a flat ~1.42 at
# 2-17 mA, so eq. (4) + Rs is fitted over 2-20 mA.
SETTINGS["9_Oct_IV_25C_GaAs_LED_Version3.csv"] = dict(
    offset_window_v=None,
    sweep_start_v=0.5,
    fit_range_a=(7.4e-4, 1.23e-3),
    window_lows_a=(6.5e-4, 7.4e-4, 8.4e-4),
    window_highs_a=(1.08e-3, 1.23e-3, 1.39e-3),
    rs_fit_range_a=(2e-3, 2e-2),
    rs_window_lows_a=(5e-4, 1e-3, 2e-3),
    rs_window_highs_a=(1e-2, 1.5e-2, 2e-2),
    device="GaAs LED",
)
# GaAs LED again (T25), sweep from 0.45 V. Below ~0.85 V there is an extra 1-2 µA of
# noisy current, ~100x the floor of Version3, that swamps the n_id ~ 2 region (e.g.
# 1.4 µA at 0.52 V, against 1.5 nA in Version3), and the local n_id is unusable below
# ~1.0 V; above that the run matches Version3 to within 1-30 %. Same fit windows as
# Version3. 35 readings (on the 1 µA range) have bogus voltages, two of them at the
# start (-0.21, -0.17 V).
SETTINGS["9_Oct_IV_25C_GaAs_LED_T25.csv"] = {
    **SETTINGS["9_Oct_IV_25C_GaAs_LED_Version3.csv"],
    "sweep_start_v": 0.45,
    "local_n_min_current_a": 1e-4,
}
# GaAs LED at 45 °C, sweep from 0.5 V. Below ~1 mA the current is swamped by an erratic
# extra 3-50 µA (18 µA at 0.5 V, against ~1 nA at 25 °C), and even at 1-10 mA the local
# n_id scatters by ±0.3, with fit residuals ~10x those at 25 °C; windows starting at
# 0.5 mA give unphysical fits. Eq. (4) is fitted over the flattest visible stretch,
# 1.1-2.7 mA (1.085-1.125 V); eq. (4) + Rs over 2-20 mA as at 25 °C, with alternative
# windows starting at 1-3 mA. 30 readings have bogus voltages.
SETTINGS["9_Oct_IV_45C_GaAs_LED_T45.csv"] = dict(
    offset_window_v=None,
    sweep_start_v=0.5,
    fit_range_a=(1.1e-3, 2.7e-3),
    window_lows_a=(1.0e-3, 1.1e-3, 1.4e-3),
    window_highs_a=(2.2e-3, 2.7e-3, 3.3e-3),
    rs_fit_range_a=(2e-3, 2e-2),
    rs_window_lows_a=(1e-3, 2e-3, 3e-3),
    rs_window_highs_a=(1e-2, 1.5e-2, 2e-2),
    local_n_min_current_a=1e-3,
    temperature_c=45,
    device="GaAs LED",
)
# GaAs LED temperature series, 296.5-360.3 K, each a sweep from 0.5 V. Readings taken on
# the 100 nA range read back a bogus -0.8 to -1.0 V; one of them comes first, so it is
# dropped by sweep_start_v. No offset correction: the sweep starts where the LED already
# passes 7-300 nA. At every temperature the local n_id is flat at ~2.0 (generation-
# recombination) from ~1 µA to 30-100 µA, then dips below 2 at ~1-2 mA before series
# resistance takes over; the minimum rises with temperature, from 1.61 at 296.5 K to
# 1.93 at 360.3 K, as recombination takes a growing share of the current. As for
# Version3, eq. (4) is fitted on the flat bottom of that dip, where the local n_id stays
# within 0.01 of its minimum (only 2 points are within 0.005 at 296.5 K); the
# alternative windows are within 0.005 and 0.02 of it.
# Per temperature (K): current (A) at which the window starts, within 0.02, 0.01, 0.005
# of the minimum, then where it ends, within 0.005, 0.01, 0.02.
GAAS_T_SERIES_WINDOWS_A = {
    296.5: (6.58e-4, 8.38e-4, 9.46e-4, 1.08e-3, 1.22e-3, 1.38e-3),  # min 1.609 at 1.110 V
    305.7: (7.39e-4, 9.28e-4, 9.28e-4, 1.33e-3, 1.49e-3, 1.67e-3),  # min 1.650 at 1.100 V
    314.7: (9.02e-4, 1.01e-3, 1.01e-3, 1.41e-3, 1.57e-3, 1.75e-3),  # min 1.692 at 1.090 V
    324.1: (9.07e-4, 1.00e-3, 1.11e-3, 1.70e-3, 1.88e-3, 2.09e-3),  # min 1.743 at 1.085 V
    333.5: (9.89e-4, 1.09e-3, 1.32e-3, 1.98e-3, 1.98e-3, 2.40e-3),  # min 1.787 at 1.080 V
    342.2: (1.08e-3, 1.30e-3, 1.42e-3, 2.08e-3, 2.28e-3, 2.50e-3),  # min 1.837 at 1.070 V
    351.4: (1.28e-3, 1.52e-3, 1.66e-3, 2.38e-3, 2.60e-3, 2.83e-3),  # min 1.884 at 1.065 V
    360.3: (1.37e-3, 1.62e-3, 1.76e-3, 2.48e-3, 2.69e-3, 3.18e-3),  # min 1.933 at 1.060 V
}
for _t, _f in zip(GAAS_T_SERIES_K, GAAS_T_SERIES_FILES):
    _w = GAAS_T_SERIES_WINDOWS_A[_t]
    SETTINGS[_f.name] = dict(
        offset_window_v=None,
        sweep_start_v=0.5,
        fit_range_a=(_w[1], _w[4]),
        window_lows_a=_w[:3],
        window_highs_a=_w[3:],
        temperature_k=_t,
        device="GaAs LED",
    )
# Lowest (offset-corrected) current in the local ideality plot (A). On 8 Oct the
# offset drifts by ~0.2 nA over the sweep (-1.3 nA at 0.5 V, -1.0 nA at 1.05 V), so
# the corrected current is only good to ~0.2 nA and n_id needs I >> that.
LOCAL_N_MIN_CURRENT_A = 2e-9

NID = r"$n_\mathrm{id}$"  # ideality factor (not n_i, the intrinsic carrier density)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e6e5e0", "#b8b7b0"


def load_iv(path):
    """Return (voltage [V], current [A]) lists from a Keithley defbuffer CSV.

    The file starts with buffer metadata; the data table begins at the row whose
    first cell is "Reading". "Reading" is the measured quantity and "Value" the
    sourced one, each followed by its unit: a current sweep (6 Oct) measures
    voltage, a voltage sweep (8 Oct) measures current.
    """
    voltage, current = [], []
    with open(path, newline="") as f:
        rows = csv.reader(f)
        for row in rows:
            if row and row[0] == "Reading":
                read_col, value_col = 0, row.index("Value")
                break
        else:
            raise ValueError(f"No 'Reading' header row found in {path}")

        for row in rows:
            if len(row) <= value_col + 1 or not row[read_col]:
                continue
            columns = {row[read_col + 1]: float(row[read_col]),
                       row[value_col + 1]: float(row[value_col])}
            voltage.append(columns["Volt DC"])
            current.append(columns["Amp DC"])
    return voltage, current


def current_offset(voltage, current, window):
    """Zero offset of the current measurement: (median, spread, N) of the current in window (V).

    Returns zero offset if there is no window or no points in it.
    """
    v, i = np.asarray(voltage), np.asarray(current)
    if window is None:
        return 0.0, 0.0, 0
    s = (v >= window[0]) & (v <= window[1])
    if not s.any():
        return 0.0, 0.0, 0
    return np.median(i[s]), np.std(i[s]), int(s.sum())


def fit_shockley(voltage, current, temperature=T_KELVIN, series_resistance=False):
    """Fit the generalised Shockley equation, eq. (4) of the lab script.

    I = I0 exp(q(V - I Rs) / (n_id kB T)) rearranges to
        V = (n_id kT/q) ln I - (n_id kT/q) ln I0 + Rs I,
    which is linear in its parameters, so ordinary least squares gives them and
    their covariance directly. With series_resistance=False, Rs = 0 and this is the
    straight line of eq. (4) in ln I. The -1 term is dropped, valid as long as
    I >> I0 in the fit window.
    """
    v, i = np.asarray(voltage), np.asarray(current)
    columns = [np.log(i), np.ones_like(i)] + ([i] if series_resistance else [])
    design = np.column_stack(columns)
    p, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ p
    cov = (resid @ resid / (len(v) - len(p))) * np.linalg.inv(design.T @ design)

    a, b = p[:2]
    vt = K_B_OVER_Q * temperature
    grad = np.array([b / a**2, -1 / a])  # d(ln I0)/d(a, b), with ln I0 = -b/a
    fit = {
        "n_id": a / vt,
        "n_id_err": np.sqrt(cov[0, 0]) / vt,
        "ln_I0": -b / a,
        "ln_I0_err": np.sqrt(grad @ cov[:2, :2] @ grad),
        "rms_V": np.sqrt(np.mean(resid**2)),
        "N": len(v),
        "T": temperature,
    }
    if series_resistance:
        fit["Rs"] = p[2]
        fit["Rs_err"] = np.sqrt(cov[2, 2])
    return fit


def local_ideality(voltage, current, temperature=T_KELVIN):
    """Local ideality factor n_id = (q/kT) dV/d(ln I): the value eq. (4) would give at each point.

    A region that obeys eq. (4) shows up as a plateau; n_id = 1 means diffusion
    dominates, n_id = 2 generation-recombination in the depletion region.
    """
    return np.gradient(voltage, np.log(current)) / (K_B_OVER_Q * temperature)


def model_voltage(current, fit):
    vt = K_B_OVER_Q * fit["T"]
    return fit["n_id"] * vt * (np.log(current) - fit["ln_I0"]) + fit.get("Rs", 0.0) * current


def window_fits(v, i, lows, highs, temperature=T_KELVIN, series_resistance=False):
    """Fits over the alternative windows (systematic uncertainty from the range choice)."""
    fits = []
    for lo in lows:
        for hi in highs:
            s = (i > lo) & (i <= hi)
            if s.sum() > 2 + series_resistance:  # need a residual degree of freedom
                fits.append(fit_shockley(v[s], i[s], temperature, series_resistance))
    return fits


def range_label(fit_range):
    """Fit window as text, in µA below 1 mA, e.g. "0.6–37.5 µA" or "1–100 mA"."""
    if fit_range[1] < 1e-3:
        return f"{fit_range[0]*1e6:g}–{fit_range[1]*1e6:g} µA"
    return f"{fit_range[0]*1e3:g}–{fit_range[1]*1e3:g} mA"


def spread(fits, key):
    values = [f[key] for f in fits]
    return min(values), max(values)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_fit(v, i, fit, fits, fit_range, title, path):
    """Measured ln I vs V with the fitted curve, and the voltage residuals below."""
    in_fit = (i > fit_range[0]) & (i <= fit_range[1])
    ln_i = np.log(i)
    with_rs = "Rs" in fit
    name = "Eq. (4) + Rₛ" if with_rs else "Eq. (4)"

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.scatter(v, ln_i, s=14, color=BLUE, zorder=3, label="Measured V")

    # Model over the whole current range (dashed), solid inside the fit window
    i_line = np.geomspace(i.min(), i.max(), 400)
    inside = (i_line > fit_range[0]) & (i_line <= fit_range[1])
    ax.plot(model_voltage(i_line, fit), np.log(i_line),
            color=INK, lw=1.5, ls="--", zorder=4, label=f"{name} extrapolated")
    ax.plot(model_voltage(i_line[inside], fit), np.log(i_line[inside]),
            color=INK, lw=1.5, zorder=5, label=f"Fit: {name}")

    vt = K_B_OVER_Q * fit["T"]
    n_lo, n_hi = spread(fits, "n_id")
    lines = [f"Fit {range_label(fit_range)}, T = {fit['T']:g} K"]
    if not with_rs:
        lines.append(f"slope q/({NID} kT) = {1 / (fit['n_id'] * vt):.1f} V⁻¹")
    lines.append(f"{NID} = {fit['n_id']:.2f} ± {fit['n_id_err']:.2f} (stat), "
                 f"{n_lo:.2f}–{n_hi:.2f} over windows")
    lines.append(f"I₀ = {np.exp(fit['ln_I0']):.1e} A  (ln I₀ = {fit['ln_I0']:.1f} ± {fit['ln_I0_err']:.1f})")
    if with_rs:
        rs_lo, rs_hi = spread(fits, "Rs")
        lines.append(f"Rₛ = {fit['Rs']:.3f} ± {fit['Rs_err']:.3f} Ω (stat), "
                     f"{rs_lo:.2f}–{rs_hi:.2f} Ω over windows")
    ax.text(0.03, 0.97, "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=9, color=INK,
            bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title(title, color=INK)
    ax.set_ylabel("ln(I / 1 A)", color=MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style(ax)

    resid_mV = (v - model_voltage(i, fit)) * 1e3
    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.scatter(v[in_fit], resid_mV[in_fit], s=10, color=BLUE, zorder=3, label="In fit")
    ax_res.scatter(v[~in_fit], resid_mV[~in_fit], s=10, color=FAINT, zorder=3, label="Not fitted")
    ax_res.set_xlabel("Voltage (V)", color=MUTED)
    ax_res.set_ylabel("V − V_fit (mV)", color=MUTED)
    ax_res.legend(loc="upper left" if resid_mV.max() > -resid_mV.min() else "lower left",
                  fontsize=8, frameon=False)
    style(ax_res)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_local_ideality(v, i, fit_range, device, temperature, temp_label, path, rs_fit=None):
    """Local ideality factor against voltage, also from the junction voltage V - I Rs if rs_fit is given."""
    in_fit = (i > fit_range[0]) & (i <= fit_range[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(v[in_fit].min(), v[in_fit].max(), color=GRID, alpha=0.6, zorder=0,
               label="Eq. (4) fit range")
    for n_ref, text in ((1, f"{NID} = 1  diffusion"), (2, f"{NID} = 2  recombination")):
        ax.axhline(n_ref, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(v.max(), n_ref + 0.15, text, ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.scatter(v, local_ideality(v, i, temperature), s=14, color=BLUE, zorder=3,
               label=f"From measured V: (q/kT) dV/d(ln I)" if rs_fit else f"(q/kT) dV/d(ln I)")
    if rs_fit:
        ax.scatter(v, local_ideality(v - i * rs_fit["Rs"], i, temperature), s=14, color=ORANGE, zorder=3,
                   label=f"From junction V − IRₛ (Rₛ = {rs_fit['Rs']:.3f} Ω)")

    ax.set_title(f"{device} local ideality factor at {temp_label}", color=INK)
    ax.set_xlabel("Voltage (V)", color=MUTED)
    ax.set_ylabel(f"Local ideality factor {NID}", color=MUTED)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def print_fit(label, fit, fits, fit_range):
    n_lo, n_hi = spread(fits, "n_id")
    print(f"\n{label}, {range_label(fit_range)} "
          f"({fit['N']} points), T = {fit['T']:g} K")
    print(f"  n_id = {fit['n_id']:.3f} ± {fit['n_id_err']:.3f} (stat); "
          f"{n_lo:.2f}–{n_hi:.2f} over alternative fit windows")
    print(f"  I0   = {np.exp(fit['ln_I0']):.2e} A  (ln I0 = {fit['ln_I0']:.2f} ± {fit['ln_I0_err']:.2f})")
    if "Rs" in fit:
        rs_lo, rs_hi = spread(fits, "Rs")
        print(f"  Rs   = {fit['Rs']:.4f} ± {fit['Rs_err']:.4f} ohm (stat); "
              f"{rs_lo:.3f}–{rs_hi:.3f} ohm over alternative fit windows")
    print(f"  RMS residual = {fit['rms_V']*1e3:.2f} mV")


def analyse(data_file):
    """Make all the plots and fits for one CSV file; return the eq. (4) fit and its window fits."""
    settings = SETTINGS.get(data_file.name, LED_SETTINGS)
    device = settings.get("device", "LED")
    temp_c = settings.get("temperature_c", 25)
    temperature = settings.get("temperature_k", temp_c + 273.15)
    temp_label = f"{temperature:g} K" if "temperature_k" in settings else f"{temp_c:g} °C"
    fit_range, offset_window = settings["fit_range_a"], settings["offset_window_v"]
    voltage, current = load_iv(data_file)

    out_dir = PLOTS_DIR / data_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def output(suffix=""):
        return out_dir / f"{data_file.stem}{suffix}.png"
    OUTPUT_FILE, LOG_OUTPUT_FILE = output(), output("_log")
    FIT_OUTPUT_FILE, LOCAL_N_OUTPUT_FILE = output("_fit"), output("_local_n")
    RS_FIT_OUTPUT_FILE = output("_fit_Rs")

    print(f"\n=== {data_file.name} ===")

    # Drop readings recorded out of sweep order: when changing current range the SMU
    # sometimes reads back a wrong voltage (8 Oct LED around 1.31 V; GaAs LED -0.62 V
    # in the middle of the sweep)
    v_all = np.array(voltage)
    in_sweep = v_all >= np.maximum.accumulate(v_all)
    if "sweep_start_v" in settings:
        in_sweep &= v_all >= settings["sweep_start_v"] - 1e-3
    if not in_sweep.all():
        print(f"Dropped {np.sum(~in_sweep)} points recorded out of sweep order (bogus readbacks)")
    voltage = list(v_all[in_sweep])
    current = list(np.array(current)[in_sweep])

    current_mA = [i * 1e3 for i in current]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(voltage, current_mA, s=14, color=BLUE, zorder=3)

    ax.set_title(f"{device} I–V Characteristic at {temp_label}", color=INK)
    ax.set_xlabel("Voltage (V)", color=MUTED)
    ax.set_ylabel("Current (mA)", color=MUTED)
    style(ax)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(f"Plotted {len(voltage)} points -> {OUTPUT_FILE}")

    # A log axis can only show I > 0
    pos = [(v, i) for v, i in zip(voltage, current_mA) if i > 0]
    v_pos, i_pos = zip(*pos)

    fig_log, ax_log = plt.subplots(figsize=(7, 5))
    ax_log.scatter(v_pos, i_pos, s=14, color=BLUE, zorder=3)
    ax_log.set_yscale("log")

    ax_log.set_title(f"{device} I–V Characteristic at {temp_label} (log current)", color=INK)
    ax_log.set_xlabel("Voltage (V)", color=MUTED)
    ax_log.set_ylabel("Current (mA, log scale)", color=MUTED)
    style(ax_log)
    ax_log.grid(True, which="minor", color=GRID, linewidth=0.5, zorder=0)

    fig_log.tight_layout()
    fig_log.savefig(LOG_OUTPUT_FILE, dpi=200)
    print(f"Plotted {len(v_pos)} points -> {LOG_OUTPUT_FILE}")

    # Shockley fit on the forward branch only, with the current offset removed
    v, i = np.array(voltage), np.array(current)
    offset, offset_spread, n_offset = current_offset(v, i, offset_window)
    if n_offset:
        print(f"\nCurrent offset = {offset*1e9:.3f} nA (spread {offset_spread*1e9:.2f} nA, "
              f"median of {n_offset} points at {offset_window[0]:g}–{offset_window[1]:g} V), "
              f"subtracted from all currents")
    else:
        print("\nNo current offset correction")
    i = i - offset
    # Up to the offset window the device passes no measurable current, and corrected
    # currents within the offset's scatter are noise, not diode current
    v_min = offset_window[1] if n_offset else 0.0
    forward = (v > v_min) & (i > 3 * offset_spread) & (i > 0)
    v, i = v[forward], i[forward]

    s = (i > fit_range[0]) & (i <= fit_range[1])
    fit = fit_shockley(v[s], i[s], temperature)
    fits = window_fits(v, i, settings["window_lows_a"], settings["window_highs_a"], temperature)
    print_fit("Shockley fit, eq. (4)", fit, fits, fit_range)

    plot_fit(v, i, fit, fits, fit_range,
             f"{device} ln(I)–V at {temp_label}: Shockley equation fit", FIT_OUTPUT_FILE)
    plots = [FIT_OUTPUT_FILE, LOCAL_N_OUTPUT_FILE]

    rs_fit = None
    if "rs_fit_range_a" in settings:
        rs_range = settings["rs_fit_range_a"]
        s = (i > rs_range[0]) & (i <= rs_range[1])
        rs_fit = fit_shockley(v[s], i[s], temperature, series_resistance=True)
        rs_fits = window_fits(v, i, settings["rs_window_lows_a"], settings["rs_window_highs_a"],
                              temperature, series_resistance=True)
        print_fit("Shockley fit with series resistance, eq. (4) + Rs", rs_fit, rs_fits, rs_range)
        plot_fit(v, i, rs_fit, rs_fits, rs_range,
                 f"{device} ln(I)–V at {temp_label}: fit with series resistance", RS_FIT_OUTPUT_FILE)
        plots.append(RS_FIT_OUTPUT_FILE)

    valid = i > settings.get("local_n_min_current_a", LOCAL_N_MIN_CURRENT_A)
    v_n, i_n = v[valid], i[valid]
    # dV/d(ln I) needs I increasing as well as V
    in_order = (v_n >= np.maximum.accumulate(v_n)) & (i_n >= np.maximum.accumulate(i_n))
    plot_local_ideality(v_n[in_order], i_n[in_order], fit_range, device, temperature, temp_label,
                        LOCAL_N_OUTPUT_FILE,
                        rs_fit)
    for path in plots:
        print(f"Plot -> {path}")
    return fit, fits


def main():
    data_files = [Path(arg) for arg in sys.argv[1:]] or DATA_FILES
    for data_file in data_files:
        analyse(data_file)
    plt.show()


if __name__ == "__main__":
    main()
