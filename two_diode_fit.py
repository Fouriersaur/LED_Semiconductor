"""Two-diode fit of the GaAs LED temperature series, to separate the diffusion current.

    I = I01 (exp(q(V - I Rs)/kB T) - 1) + I02 (exp(q(V - I Rs)/(n2 kB T)) - 1)

The n_id = 1 term is the diffusion (Shockley) current, so I01 is the diffusion
saturation current that eqs. (2) and (3) of the lab script describe; the second term
is generation-recombination in the depletion region (n2 = 2) and Rs the series
resistance. I01 is then fitted with eq. (3), I01 = c T^3 exp(-Eg/(kB T)).

Each curve is fitted four times, over two current ranges, each with
  * n2 = 2 fixed, the standard two-diode model;
  * n2 free, as a check: the n_id ~ 2 plateau of this LED is at 2.03-2.07, not 2.0,
    so with n2 fixed the slope mismatch has to be absorbed by I01 and Rs. This matters
    above ~340 K, where diffusion is only a small share of the current.
The ranges are 10 µA-20 mA, which leaves out the low-current part of the plateau where
n2 = 2 fits worst, and the full sweep (first reading, a few nA to ~1 µA, to 20 mA).

Like the eq. (4) fits, the fit minimises voltage residuals: for a given I, the junction
voltage Vj = V - I Rs follows from I + I01 + I02 = I01 exp(Vj/vt) + I02 exp(Vj/(n2 vt)),
solved by Newton's method, so the model gives V(I) directly.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from I0_vs_T import fit_arrhenius, model_ln_i0
from IV_Curve_25C import (
    BLUE, FAINT, GAAS_T_SERIES_FILES, GAAS_T_SERIES_K, GRID, INK, K_B_OVER_Q, MUTED, ORANGE,
    PLOTS_DIR, SETTINGS, load_iv, style,
)

# Fit ranges (A): main window, then the alternative window starts and ends (systematic
# error from the range). The sweeps end at the 20 mA source limit. Below ~10 µA the
# plateau's n_id (2.03-2.07) differs enough from 2 to bias the n2 = 2 fit; "full" starts
# at the first reading, with alternatives starting at 0.1 and 1 µA.
FIT_RANGES = {
    "10uA": dict(fit_range_a=(1e-5, 2e-2), window_lows_a=(3e-6, 1e-5, 3e-5),
                 window_highs_a=(1e-2, 1.5e-2, 2e-2), suffix="", label="10 µA – 20 mA"),
    "full": dict(fit_range_a=(0.0, 2e-2), window_lows_a=(0.0, 1e-7, 1e-6),
                 window_highs_a=(1e-2, 1.5e-2, 2e-2), suffix="_full", label="full sweep, to 20 mA"),
}

I0_PLOTS_DIR = PLOTS_DIR / "9_Oct_IV_GaAs_LED_I0_vs_T"
GREEN = "#2e9e5b"


def load_sweep(data_file):
    """(V, I) arrays of the forward sweep, with bogus out-of-order readbacks dropped as in analyse()."""
    v, i = map(np.array, load_iv(data_file))
    keep = (v >= np.maximum.accumulate(v)) & (i > 0)
    if "sweep_start_v" in SETTINGS[data_file.name]:
        keep &= v >= SETTINGS[data_file.name]["sweep_start_v"] - 1e-3
    return v[keep], i[keep]


def junction_voltage(i, i01, i02, n2, vt):
    """Vj with I + I01 + I02 = I01 exp(Vj/vt) + I02 exp(Vj/(n2 vt)).

    Newton's method on ln(right side) - ln(left side), which is convex in Vj. Starting
    from the smaller of the two single-diode solutions, which lies above the root,
    it converges monotonically.
    """
    total = i + i01 + i02
    x = np.minimum(vt * np.log(total / i01), n2 * vt * np.log(total / i02))
    for _ in range(100):
        a, b = i01 * np.exp(x / vt), i02 * np.exp(x / (n2 * vt))
        step = (np.log(a + b) - np.log(total)) * (a + b) / (a / vt + b / (n2 * vt))
        x = x - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return x


def model_voltage(i, fit):
    vt = K_B_OVER_Q * fit["T"]
    vj = junction_voltage(i, np.exp(fit["ln_I01"]), np.exp(fit["ln_I02"]), fit["n2"], vt)
    return vj + i * fit["Rs"]


def fit_two_diode(v, i, temperature, n2=2.0):
    """Least-squares two-diode fit of V(I); n2=None fits n2 too.

    Parameters are ln I01, ln I02, Rs (and n2). Starting values: I02 from the n_id ~ 2
    plateau at ~10 µA, I01 from half the current at ~1 mA (the n_id dip), Rs = 3 Ω.
    """
    vt = K_B_OVER_Q * temperature
    j2, j1 = np.argmin(abs(i - 1e-5)), np.argmin(abs(i - 1e-3))
    p0 = [np.log(i[j1] / 2) - v[j1] / vt, np.log(i[j2]) - v[j2] / (2 * vt), 3.0]
    if n2 is None:
        p0.append(2.0)

    def unpack(p):
        return dict(ln_I01=p[0], ln_I02=p[1], Rs=p[2], n2=p[3] if n2 is None else n2, T=temperature)

    result = least_squares(lambda p: model_voltage(i, unpack(p)) - v, p0)
    resid, jac = result.fun, result.jac
    cov = (resid @ resid / (len(v) - len(p0))) * np.linalg.inv(jac.T @ jac)
    fit = unpack(result.x)
    fit.update(ln_I01_err=np.sqrt(cov[0, 0]), ln_I02_err=np.sqrt(cov[1, 1]),
               Rs_err=np.sqrt(cov[2, 2]), n2_err=np.sqrt(cov[3, 3]) if n2 is None else 0.0,
               rms_V=np.sqrt(np.mean(resid**2)), N=len(v))
    return fit


def fit_with_windows(v, i, temperature, n2, fit_range):
    """Main fit, plus fits over the alternative windows (systematic error from the range)."""
    lo, hi = fit_range["fit_range_a"]
    s = (i > lo) & (i <= hi)
    fit = fit_two_diode(v[s], i[s], temperature, n2)
    fit["range"] = fit_range
    fits = []
    for lo in fit_range["window_lows_a"]:
        for hi in fit_range["window_highs_a"]:
            s = (i > lo) & (i <= hi)
            fits.append(fit_two_diode(v[s], i[s], temperature, n2))
    lo, hi = min(f["ln_I01"] for f in fits), max(f["ln_I01"] for f in fits)
    # Statistical error and half the spread over the alternative windows, in quadrature
    fit["ln_I01_tot_err"] = np.hypot(fit["ln_I01_err"], (hi - lo) / 2)
    lo, hi = min(f["ln_I02"] for f in fits), max(f["ln_I02"] for f in fits)
    fit["ln_I02_tot_err"] = np.hypot(fit["ln_I02_err"], (hi - lo) / 2)
    fit["Rs_range"] = (min(f["Rs"] for f in fits), max(f["Rs"] for f in fits))
    return fit


def plot_two_diode(v, i, fit, title, path):
    """Measured ln I vs V with the two-diode fit and its two components, and residuals."""
    vt = K_B_OVER_Q * fit["T"]
    lo, hi = fit["range"]["fit_range_a"]
    in_fit = (i > lo) & (i <= hi)
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.scatter(v, np.log(i), s=14, color=BLUE, zorder=3, label="Measured")

    i_line = np.geomspace(i.min(), i.max(), 400)
    v_line = model_voltage(i_line, fit)
    vj = v_line - i_line * fit["Rs"]
    i_diff = np.exp(fit["ln_I01"]) * np.expm1(vj / vt)
    i_rec = np.exp(fit["ln_I02"]) * np.expm1(vj / (fit["n2"] * vt))
    ax.plot(v_line, np.log(i_line), color=INK, lw=1.5, zorder=4, label="Two-diode fit, total")
    ax.plot(v_line, np.log(i_diff), color=ORANGE, lw=1.3, ls="--", zorder=4,
            label="Diffusion, I₀₁ term (n = 1)")
    ax.plot(v_line, np.log(i_rec), color=GREEN, lw=1.3, ls="--", zorder=4,
            label=f"Recombination, I₀₂ term (n = {fit['n2']:.2f})")
    ax.set_ylim(np.log(i.min()) - 1, np.log(i.max()) + 1)

    lines = [f"Fit {fit['range']['label']}, T = {fit['T']:g} K",
             f"I₀₁ = {np.exp(fit['ln_I01']):.2e} A  (ln I₀₁ = {fit['ln_I01']:.2f} ± {fit['ln_I01_tot_err']:.2f})",
             f"I₀₂ = {np.exp(fit['ln_I02']):.2e} A  (ln I₀₂ = {fit['ln_I02']:.2f} ± {fit['ln_I02_tot_err']:.2f})",
             f"Rₛ = {fit['Rs']:.2f} ± {fit['Rs_err']:.2f} Ω (stat), "
             f"{fit['Rs_range'][0]:.2f}–{fit['Rs_range'][1]:.2f} Ω over windows",
             f"RMS residual = {fit['rms_V']*1e3:.2f} mV"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title(title, color=INK)
    ax.set_ylabel("ln(I / 1 A)", color=MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style(ax)

    resid_mV = (v - model_voltage(i, fit)) * 1e3
    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.scatter(v[in_fit], resid_mV[in_fit], s=10, color=BLUE, zorder=3, label="In fit")
    ax_res.scatter(v[~in_fit], resid_mV[~in_fit], s=10, color=FAINT, zorder=3, label="Not fitted")
    ax_res.set_ylim(-max(5, 1.5 * np.abs(resid_mV[in_fit]).max()),
                    max(5, 1.5 * np.abs(resid_mV[in_fit]).max()))
    ax_res.set_xlabel("Voltage (V)", color=MUTED)
    ax_res.set_ylabel("V − V_fit (mV)", color=MUTED)
    ax_res.legend(loc="upper left", fontsize=8, frameon=False)
    style(ax_res)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_arrhenius(label, T, ln_i0, ln_i0_err, t_power, eg_factor=1):
    p, cov, chi2_dof = fit_arrhenius(T, ln_i0, ln_i0_err, t_power)
    eg, eg_err = eg_factor * p[1], eg_factor * np.sqrt(cov[1, 1])
    print(f"  {label}: Eg = {eg:.3f} ± {eg_err:.3f} eV, ln c = {p[0]:.2f} ± {np.sqrt(cov[0, 0]):.2f}, "
          f"chi2/dof = {chi2_dof:.1f}")
    return p, cov, chi2_dof


def column(fits, key):
    return np.array([f[key] for f in fits])


def plot_i01_vs_t(T, results, fit_range, path):
    """I01 against T for the n2 = 2 and n2 free fits over one range, each with its eq. (3) fit."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    t_line = np.linspace(T.min() - 3, T.max() + 3, 300)
    for (fits, p, cov, chi2), color, name, dx in zip(
        results, (BLUE, ORANGE), ("n₂ = 2", "n₂ free"), (-0.6, 0.6)
    ):
        ln_i01, err = column(fits, "ln_I01"), column(fits, "ln_I01_tot_err")
        i01 = np.exp(ln_i01)
        # Asymmetric error bars on the log axis: exp(ln I01 ± err)
        ax.errorbar(T + dx, i01, yerr=[i01 - np.exp(ln_i01 - err), np.exp(ln_i01 + err) - i01],
                    fmt="o", ms=5, color=color, ecolor=color, capsize=3, zorder=3,
                    label=f"I₀₁, two-diode fit with {name}")
        ax.plot(t_line, np.exp(model_ln_i0(t_line, p, 3)), color=color, lw=1.4, zorder=2,
                label=f"Eq. (3) fit: $E_g$ = {p[1]:.2f} ± {np.sqrt(cov[1, 1]):.2f} eV, "
                      f"χ²/dof = {chi2:.1f}")
    ax.set_yscale("log")
    ax.set_title(f"GaAs LED diffusion saturation current I₀₁ against temperature\n"
                 f"(two-diode fit, {fit_range['label']})", color=INK)
    ax.set_xlabel("Temperature (K)", color=MUTED)
    ax.set_ylabel("I₀₁ (A, log scale)", color=MUTED)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    style(ax)
    ax.grid(True, which="minor", color=GRID, linewidth=0.5, zorder=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Plot -> {path}")


def main():
    T = np.array(GAAS_T_SERIES_K)
    sweeps = [load_sweep(data_file) for data_file in GAAS_T_SERIES_FILES]
    for fit_range in FIT_RANGES.values():
        suffix = fit_range["suffix"]
        fixed, free = [], []
        for t, data_file, (v, i) in zip(T, GAAS_T_SERIES_FILES, sweeps):
            fixed.append(fit_with_windows(v, i, t, 2.0, fit_range))
            free.append(fit_with_windows(v, i, t, None, fit_range))
            out_dir = PLOTS_DIR / data_file.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_two_diode(v, i, fixed[-1], f"GaAs LED ln(I)–V at {t:g} K: two-diode fit (n₂ = 2)",
                           out_dir / f"{data_file.stem}_fit_two_diode{suffix}.png")
            plot_two_diode(v, i, free[-1], f"GaAs LED ln(I)–V at {t:g} K: two-diode fit (n₂ free)",
                           out_dir / f"{data_file.stem}_fit_two_diode{suffix}_n2_free.png")

        for name, fits in (("n2 = 2", fixed), ("n2 free", free)):
            print(f"\n=== Two-diode fit, {name}, {fit_range['label']} ===")
            print("   T (K)  N    I01 (A)    ln I01           I02 (A)    Rs (ohm)            n2      RMS (mV)")
            for f in fits:
                print(f"  {f['T']:6.1f} {f['N']:3d}  {np.exp(f['ln_I01']):.2e}  "
                      f"{f['ln_I01']:7.2f} ± {f['ln_I01_tot_err']:4.2f}  {np.exp(f['ln_I02']):.2e}  "
                      f"{f['Rs']:.2f} ({f['Rs_range'][0]:.2f}–{f['Rs_range'][1]:.2f})  "
                      f"{f['n2']:.3f}  {f['rms_V']*1e3:.2f}")

        print(f"\nEq. (3), I01 = c T^3 exp(-Eg/kT), {fit_range['label']}:")
        results = [(fits, *print_arrhenius(name, T, column(fits, "ln_I01"),
                                           column(fits, "ln_I01_tot_err"), 3))
                   for name, fits in (("n2 = 2 ", fixed), ("n2 free", free))]
        print("Recombination term (n2 = 2), I02 = c' T^(3/2) exp(-Eg/2kT):")
        print_arrhenius("n2 = 2 ", T, column(fixed, "ln_I02"), column(fixed, "ln_I02_tot_err"), 1.5, 2)

        plot_i01_vs_t(T, results, fit_range,
                      I0_PLOTS_DIR / f"9_Oct_IV_GaAs_LED_I01_vs_T_two_diode{suffix}.png")


if __name__ == "__main__":
    main()
