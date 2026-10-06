"""Plot and fit the I-V characteristic of an LED at 25 °C from a Keithley SMU buffer export."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = Path(__file__).with_name("defbuffer1_1006_202917(in).csv")
OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C.png")
LOG_OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C_log.png")
FIT_OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C_fit.png")
RS_FIT_OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C_fit_Rs.png")
JUNCTION_OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C_junction.png")
LOCAL_N_OUTPUT_FILE = Path(__file__).with_name("IV_Curve_25C_local_n.png")

T_KELVIN = 298.15  # 25 °C
K_B_OVER_Q = 8.617333262e-5  # V/K

# The first point (80 µA, straight after the 0 A step) is an outlier.
MIN_CURRENT_A = 0.15e-3
# Eq. (4) fit window (A): the only range where the measured local ideality factor
# is close to a textbook value (n_id ≈ 2).
FIT_RANGE_A = (MIN_CURRENT_A, 0.55e-3)
# Eq. (4) + R_s fit window (A). Above ~7 mA the measured V falls increasingly below
# the model (constant n_id, R_s and T no longer hold, e.g. self-heating).
RS_FIT_RANGE_A = (MIN_CURRENT_A, 7e-3)
# Alternative windows, used to estimate the systematic error from the choice of range
WINDOW_LOWS_A = (0.0, MIN_CURRENT_A, 0.25e-3)
WINDOW_HIGHS_A = (0.45e-3, 0.55e-3, 0.65e-3)
RS_WINDOW_HIGHS_A = (3e-3, 5e-3, 7e-3)

NID = r"$n_\mathrm{id}$"  # ideality factor (not n_i, the intrinsic carrier density)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e6e5e0", "#b8b7b0"


def load_iv(path):
    """Return (voltage [V], current [A]) lists from a Keithley defbuffer CSV.

    The file starts with buffer metadata; the data table begins at the row whose
    first cell is "Reading". "Reading" is the measured voltage and "Value" is the
    sourced current.
    """
    voltage, current = [], []
    with open(path, newline="") as f:
        rows = csv.reader(f)
        for row in rows:
            if row and row[0] == "Reading":
                v_col, i_col = 0, row.index("Value")
                break
        else:
            raise ValueError(f"No 'Reading' header row found in {path}")

        for row in rows:
            if len(row) <= i_col or not row[v_col]:
                continue
            voltage.append(float(row[v_col]))
            current.append(float(row[i_col]))
    return voltage, current


def fit_shockley(voltage, current, temperature=T_KELVIN, series_resistance=False):
    """Fit the generalised Shockley equation, eq. (4) of the lab script.

    I = I0 exp(q(V - I R_s) / (n_id kB T)) rearranges to
        V = (n_id kT/q) ln I - (n_id kT/q) ln I0 + R_s I,
    which is linear in its parameters, so ordinary least squares gives them and
    their covariance directly. With series_resistance=False, R_s = 0 and this is
    the straight line of eq. (4) in ln I. V is the dependent variable because the
    current was sourced (precise) and the voltage measured (noisy). The -1 term is
    dropped, valid since I >> I0 everywhere in the data.
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


def junction_voltage(voltage, current, fit):
    return voltage - current * fit.get("Rs", 0.0)


def model_voltage(current, fit, temperature=T_KELVIN):
    vt = K_B_OVER_Q * temperature
    return fit["n_id"] * vt * (np.log(current) - fit["ln_I0"]) + fit.get("Rs", 0.0) * current


def window_fits(v, i, highs, series_resistance=False):
    """Fits over the alternative windows (systematic uncertainty from the range choice)."""
    fits = []
    for lo in WINDOW_LOWS_A:
        for hi in highs:
            s = (i > lo) & (i <= hi)
            if s.sum() > 2 + series_resistance:  # need a residual degree of freedom
                fits.append(fit_shockley(v[s], i[s], series_resistance=series_resistance))
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
    with_rs = "Rs" in fit
    name = "Eq. (4) + Rₛ" if with_rs else "Eq. (4)"

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.scatter(v, ln_i, s=14, color=BLUE, zorder=3, label="Measured V")

    i_line = np.geomspace(i.min(), i.max(), 400)
    inside = i_line <= fit_range[1]
    for mask, ls in ((inside, "-"), (~inside, "--")):
        ax.plot(model_voltage(i_line[mask], fit), np.log(i_line[mask]),
                color=INK, lw=1.5, ls=ls, zorder=4,
                label=f"Fit: {name}" if ls == "-" else f"{name} extrapolated")

    vt = K_B_OVER_Q * T_KELVIN
    n_lo, n_hi = spread(fits, "n_id")
    lines = [f"Fit {fit_range[0]*1e3:g}–{fit_range[1]*1e3:g} mA, T = {T_KELVIN} K"]
    if not with_rs:
        lines.append(f"slope q/({NID} kT) = {1 / (fit['n_id'] * vt):.1f} V⁻¹")
    lines.append(f"{NID} = {fit['n_id']:.2f} ± {fit['n_id_err']:.2f} (stat), "
                 f"{n_lo:.1f}–{n_hi:.1f} over windows")
    lines.append(f"I₀ = {np.exp(fit['ln_I0']):.1e} A  (ln I₀ = {fit['ln_I0']:.1f} ± {fit['ln_I0_err']:.1f})")
    if with_rs:
        rs_lo, rs_hi = spread(fits, "Rs")
        lines.append(f"Rₛ = {fit['Rs']:.1f} ± {fit['Rs_err']:.1f} Ω (stat), "
                     f"{rs_lo:.0f}–{rs_hi:.0f} Ω over windows")
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


def plot_junction(v, i, fit, fit_range, path):
    """ln I against the junction voltage V - I R_s, with the straight line of eq. (4)."""
    vt = K_B_OVER_Q * T_KELVIN

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(junction_voltage(v, i, fit), np.log(i), s=14, color=ORANGE, zorder=3,
               label="Junction voltage V − IRₛ")

    ln_line = np.log(np.geomspace(i.min(), i.max(), 400))
    inside = ln_line <= np.log(fit_range[1])
    for mask, ls in ((inside, "-"), (~inside, "--")):
        ax.plot(fit["n_id"] * vt * (ln_line[mask] - fit["ln_I0"]), ln_line[mask],
                color=INK, lw=1.5, ls=ls, zorder=4,
                label=f"Eq. (4): ln I = ln I₀ + qV/({NID} kT)" if ls == "-" else "Eq. (4) extrapolated")

    ax.text(0.03, 0.97,
            f"Rₛ = {fit['Rs']:.1f} Ω subtracted (fit {fit_range[0]*1e3:g}–{fit_range[1]*1e3:g} mA)\n"
            f"slope q/({NID} kT) = {1 / (fit['n_id'] * vt):.1f} V⁻¹  →  {NID} = {fit['n_id']:.2f}\n"
            f"intercept ln I₀ = {fit['ln_I0']:.1f} ± {fit['ln_I0_err']:.1f}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK,
            bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title("LED ln(I) vs junction voltage at 25 °C", color=INK)
    ax.set_xlabel("Junction voltage V − IRₛ (V)", color=MUTED)
    ax.set_ylabel("ln(I / 1 A)", color=MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_local_ideality(v, i, rs_fit, path):
    """Local ideality factor against current, from the measured and the junction voltage."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(FIT_RANGE_A[0] * 1e3, FIT_RANGE_A[1] * 1e3, color=GRID, alpha=0.6, zorder=0,
               label="Eq. (4) fit range")
    for n_ref, text in ((1, f"{NID} = 1  diffusion"), (2, f"{NID} = 2  recombination")):
        ax.axhline(n_ref, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(i.max() * 1e3, n_ref + 0.15, text, ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.scatter(i * 1e3, local_ideality(v, i), s=14, color=BLUE, zorder=3,
               label=f"From measured V: (q/kT) dV/d(ln I)")
    ax.scatter(i * 1e3, local_ideality(junction_voltage(v, i, rs_fit), i), s=14, color=ORANGE,
               zorder=3, label=f"From junction V − IRₛ (Rₛ = {rs_fit['Rs']:.1f} Ω)")

    ax.set_xscale("log")
    ax.set_title("LED local ideality factor at 25 °C", color=INK)
    ax.set_xlabel("Current (mA, log scale)", color=MUTED)
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
    if "Rs" in fit:
        rs_lo, rs_hi = spread(fits, "Rs")
        print(f"  Rs   = {fit['Rs']:.2f} ± {fit['Rs_err']:.2f} ohm (stat); "
              f"{rs_lo:.1f}–{rs_hi:.1f} ohm over alternative fit windows")
    print(f"  RMS residual = {fit['rms_V']*1e3:.2f} mV")


def main():
    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_FILE
    voltage, current = load_iv(data_file)
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

    # Shockley fits on the forward branch only
    v, i = np.array(voltage), np.array(current)
    forward = (v > 0) & (i > 0)
    v, i = v[forward], i[forward]

    s = (i > FIT_RANGE_A[0]) & (i <= FIT_RANGE_A[1])
    fit = fit_shockley(v[s], i[s])
    fits = window_fits(v, i, WINDOW_HIGHS_A)
    print_fit("Shockley fit, eq. (4)", fit, fits, FIT_RANGE_A)

    s = (i > RS_FIT_RANGE_A[0]) & (i <= RS_FIT_RANGE_A[1])
    rs_fit = fit_shockley(v[s], i[s], series_resistance=True)
    rs_fits = window_fits(v, i, RS_WINDOW_HIGHS_A, series_resistance=True)
    print_fit("Shockley fit with series resistance, eq. (4) + Rs", rs_fit, rs_fits, RS_FIT_RANGE_A)

    plot_fit(v, i, fit, fits, FIT_RANGE_A,
             "LED ln(I)–V at 25 °C: Shockley equation fit", FIT_OUTPUT_FILE)
    plot_fit(v, i, rs_fit, rs_fits, RS_FIT_RANGE_A,
             "LED ln(I)–V at 25 °C: Shockley fit with series resistance", RS_FIT_OUTPUT_FILE)
    plot_junction(v, i, rs_fit, RS_FIT_RANGE_A, JUNCTION_OUTPUT_FILE)
    valid = i > MIN_CURRENT_A
    plot_local_ideality(v[valid], i[valid], rs_fit, LOCAL_N_OUTPUT_FILE)
    for path in (FIT_OUTPUT_FILE, RS_FIT_OUTPUT_FILE, JUNCTION_OUTPUT_FILE, LOCAL_N_OUTPUT_FILE):
        print(f"Plot -> {path}")
    plt.show()


if __name__ == "__main__":
    main()
