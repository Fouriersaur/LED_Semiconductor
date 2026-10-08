"""Plot and fit the I-V characteristic of an LED at 25 °C from a Keithley SMU buffer export."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILES = [
    # Path(__file__).with_name("6_Oct_IV_25C.csv"),  # current sweep, voltage measured
    Path(__file__).with_name("8_Oct_IV_25C.csv"),  # voltage sweep, current measured
    Path(__file__).with_name("8_Oct_IV_25C_Version2.csv"),  # voltage sweep, current measured
    Path(__file__).with_name("8_Oct_IV_25C_GaAs_LED.csv"),  # voltage sweep, current measured
    Path(__file__).with_name("8_Oct_IV_25C_GaAs_LED_Version2.csv"),  # voltage sweep, current measured
]
# Plots go to plots/<csv name>/, e.g. plots/8_Oct_IV_25C/8_Oct_IV_25C_fit.png
PLOTS_DIR = Path(__file__).with_name("plots")

T_KELVIN = 298.15  # 25 °C
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
# Lowest (offset-corrected) current in the local ideality plot (A). On 8 Oct the
# offset drifts by ~0.2 nA over the sweep (-1.3 nA at 0.5 V, -1.0 nA at 1.05 V), so
# the corrected current is only good to ~0.2 nA and n_id needs I >> that.
LOCAL_N_MIN_CURRENT_A = 2e-9

NID = r"$n_\mathrm{id}$"  # ideality factor (not n_i, the intrinsic carrier density)

BLUE = "#2a78d6"
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


def fit_shockley(voltage, current, temperature=T_KELVIN):
    """Fit the generalised Shockley equation, eq. (4) of the lab script.

    I = I0 exp(qV / (n_id kB T)) rearranges to
        V = (n_id kT/q) ln I - (n_id kT/q) ln I0,
    a straight line in ln I, so ordinary least squares gives the parameters and
    their covariance directly. V is the dependent variable because the current was
    sourced (precise) and the voltage measured (noisy). The -1 term is dropped,
    valid as long as I >> I0 in the fit window.
    """
    v, i = np.asarray(voltage), np.asarray(current)
    design = np.column_stack([np.log(i), np.ones_like(i)])
    p, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ p
    cov = (resid @ resid / (len(v) - len(p))) * np.linalg.inv(design.T @ design)

    a, b = p
    vt = K_B_OVER_Q * temperature
    grad = np.array([b / a**2, -1 / a])  # d(ln I0)/d(a, b), with ln I0 = -b/a
    return {
        "n_id": a / vt,
        "n_id_err": np.sqrt(cov[0, 0]) / vt,
        "ln_I0": -b / a,
        "ln_I0_err": np.sqrt(grad @ cov @ grad),
        "rms_V": np.sqrt(np.mean(resid**2)),
        "N": len(v),
    }


def local_ideality(voltage, current, temperature=T_KELVIN):
    """Local ideality factor n_id = (q/kT) dV/d(ln I): the value eq. (4) would give at each point.

    A region that obeys eq. (4) shows up as a plateau; n_id = 1 means diffusion
    dominates, n_id = 2 generation-recombination in the depletion region.
    """
    return np.gradient(voltage, np.log(current)) / (K_B_OVER_Q * temperature)


def model_voltage(current, fit, temperature=T_KELVIN):
    vt = K_B_OVER_Q * temperature
    return fit["n_id"] * vt * (np.log(current) - fit["ln_I0"])


def window_fits(v, i, lows, highs):
    """Fits over the alternative windows (systematic uncertainty from the range choice)."""
    fits = []
    for lo in lows:
        for hi in highs:
            s = (i > lo) & (i <= hi)
            if s.sum() > 2:  # need a residual degree of freedom
                fits.append(fit_shockley(v[s], i[s]))
    return fits


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

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.scatter(v, ln_i, s=14, color=BLUE, zorder=3, label="Measured V")

    i_line = np.geomspace(i.min(), i.max(), 400)
    inside = i_line <= fit_range[1]
    for mask, ls in ((inside, "-"), (~inside, "--")):
        ax.plot(model_voltage(i_line[mask], fit), np.log(i_line[mask]),
                color=INK, lw=1.5, ls=ls, zorder=4,
                label="Fit: Eq. (4)" if ls == "-" else "Eq. (4) extrapolated")

    vt = K_B_OVER_Q * T_KELVIN
    n_lo, n_hi = spread(fits, "n_id")
    lines = [
        f"Fit {fit_range[0]*1e3:g}–{fit_range[1]*1e3:g} mA, T = {T_KELVIN} K",
        f"slope q/({NID} kT) = {1 / (fit['n_id'] * vt):.1f} V⁻¹",
        f"{NID} = {fit['n_id']:.2f} ± {fit['n_id_err']:.2f} (stat), "
        f"{n_lo:.1f}–{n_hi:.1f} over windows",
        f"I₀ = {np.exp(fit['ln_I0']):.1e} A  (ln I₀ = {fit['ln_I0']:.1f} ± {fit['ln_I0_err']:.1f})",
    ]
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


def plot_local_ideality(v, i, fit_range, path):
    """Local ideality factor against voltage."""
    in_fit = (i > fit_range[0]) & (i <= fit_range[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(v[in_fit].min(), v[in_fit].max(), color=GRID, alpha=0.6, zorder=0,
               label="Eq. (4) fit range")
    for n_ref, text in ((1, f"{NID} = 1  diffusion"), (2, f"{NID} = 2  recombination")):
        ax.axhline(n_ref, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(v.max(), n_ref + 0.15, text, ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.scatter(v, local_ideality(v, i), s=14, color=BLUE, zorder=3,
               label=f"(q/kT) dV/d(ln I)")

    ax.set_title("LED local ideality factor at 25 °C", color=INK)
    ax.set_xlabel("Voltage (V)", color=MUTED)
    ax.set_ylabel(f"Local ideality factor {NID}", color=MUTED)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def print_fit(label, fit, fits, fit_range):
    n_lo, n_hi = spread(fits, "n_id")
    print(f"\n{label}, {fit_range[0]*1e3:g}–{fit_range[1]*1e3:g} mA "
          f"({fit['N']} points), T = {T_KELVIN} K")
    print(f"  n_id = {fit['n_id']:.3f} ± {fit['n_id_err']:.3f} (stat); "
          f"{n_lo:.2f}–{n_hi:.2f} over alternative fit windows")
    print(f"  I0   = {np.exp(fit['ln_I0']):.2e} A  (ln I0 = {fit['ln_I0']:.2f} ± {fit['ln_I0_err']:.2f})")
    print(f"  RMS residual = {fit['rms_V']*1e3:.2f} mV")


def analyse(data_file):
    """Make all the plots and fits for one CSV file."""
    settings = SETTINGS.get(data_file.name, LED_SETTINGS)
    fit_range, offset_window = settings["fit_range_a"], settings["offset_window_v"]
    voltage, current = load_iv(data_file)

    out_dir = PLOTS_DIR / data_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def output(suffix=""):
        return out_dir / f"{data_file.stem}{suffix}.png"
    OUTPUT_FILE, LOG_OUTPUT_FILE = output(), output("_log")
    FIT_OUTPUT_FILE, LOCAL_N_OUTPUT_FILE = output("_fit"), output("_local_n")

    print(f"\n=== {data_file.name} ===")

    # Drop readings recorded out of sweep order: when changing current range the SMU
    # sometimes reads back a wrong voltage (8 Oct LED around 1.31 V; GaAs LED -0.62 V
    # in the middle of the sweep)
    v_all = np.array(voltage)
    in_sweep = v_all >= np.maximum.accumulate(v_all)
    if not in_sweep.all():
        print(f"Dropped {np.sum(~in_sweep)} points recorded out of sweep order")
    voltage = list(v_all[in_sweep])
    current = list(np.array(current)[in_sweep])

    current_mA = [i * 1e3 for i in current]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(voltage, current_mA, s=14, color=BLUE, zorder=3)

    ax.set_title("LED I–V Characteristic at 25 °C", color=INK)
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

    ax_log.set_title("LED I–V Characteristic at 25 °C (log current)", color=INK)
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
    fit = fit_shockley(v[s], i[s])
    fits = window_fits(v, i, settings["window_lows_a"], settings["window_highs_a"])
    print_fit("Shockley fit, eq. (4)", fit, fits, fit_range)

    plot_fit(v, i, fit, fits, fit_range,
             "LED ln(I)–V at 25 °C: Shockley equation fit", FIT_OUTPUT_FILE)
    valid = i > LOCAL_N_MIN_CURRENT_A
    v_n, i_n = v[valid], i[valid]
    # dV/d(ln I) needs I increasing as well as V
    in_order = (v_n >= np.maximum.accumulate(v_n)) & (i_n >= np.maximum.accumulate(i_n))
    plot_local_ideality(v_n[in_order], i_n[in_order], fit_range, LOCAL_N_OUTPUT_FILE)
    for path in (FIT_OUTPUT_FILE, LOCAL_N_OUTPUT_FILE):
        print(f"Plot -> {path}")


def main():
    data_files = [Path(arg) for arg in sys.argv[1:]] or DATA_FILES
    for data_file in data_files:
        analyse(data_file)
    plt.show()


if __name__ == "__main__":
    main()
