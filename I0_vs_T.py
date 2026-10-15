"""Fit I0(T) of the GaAs LED temperature series with eq. (3) of the lab script.

Runs the eq. (4) fit of IV_Curve_25C.py on each file of the series (which also makes
its four plots), then fits

    I0 = c T^3 exp(-Eg / (kB T))                                    (3)

as ln I0 - 3 ln T = ln c - Eg / (kB T), which is linear in (ln c, Eg).
"""

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import (
    BLUE, FAINT, GAAS_T_SERIES_FILES, GAAS_T_SERIES_K, GRID, INK, K_B_OVER_Q, MUTED, NID,
    PLOTS_DIR, analyse, style,
)

OUTPUT_FILE = PLOTS_DIR / "9_Oct_IV_GaAs_LED_I0_vs_T" / "9_Oct_IV_GaAs_LED_I0_vs_T.png"


def fit_arrhenius(T, ln_i0, ln_i0_err, t_power):
    """Weighted least squares of ln I0 = ln c + t_power ln T - Eg/(kB T).

    Returns (ln c, Eg [eV]), their covariance, and chi^2 per degree of freedom. The
    covariance is scaled up by chi^2/dof when that exceeds 1, i.e. when the scatter
    between temperatures is larger than the per-point errors account for.
    """
    y = ln_i0 - t_power * np.log(T)
    design = np.column_stack([np.ones_like(T), -1 / (K_B_OVER_Q * T)])
    w = 1 / ln_i0_err**2
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    p = cov @ design.T @ (w * y)
    chi2_dof = np.sum(w * (y - design @ p) ** 2) / (len(T) - 2)
    return p, cov * max(chi2_dof, 1.0), chi2_dof


def model_ln_i0(T, p, t_power):
    return p[0] + t_power * np.log(T) - p[1] / (K_B_OVER_Q * T)


def main():
    T = np.array(GAAS_T_SERIES_K)
    rows = []
    for data_file in GAAS_T_SERIES_FILES:
        fit, fits = analyse(data_file)
        plt.close("all")
        lo, hi = min(f["ln_I0"] for f in fits), max(f["ln_I0"] for f in fits)
        # Statistical error and half the spread over the alternative fit windows, in quadrature
        rows.append((fit["ln_I0"], np.hypot(fit["ln_I0_err"], (hi - lo) / 2), fit["n_id"]))
    ln_i0, ln_i0_err, n_id = map(np.array, zip(*rows))

    print("\n=== I0 against temperature ===")
    print("   T (K)   n_id    I0 (A)      ln I0")
    for t, n, l, e in zip(T, n_id, ln_i0, ln_i0_err):
        print(f"  {t:6.1f}  {n:.3f}  {np.exp(l):.3e}  {l:7.2f} ± {e:.2f}")

    p, cov, chi2_dof = fit_arrhenius(T, ln_i0, ln_i0_err, 3)
    eg, eg_err = p[1], np.sqrt(cov[1, 1])
    print(f"\nEq. (3), I0 = c T^3 exp(-Eg/kT):")
    print(f"  Eg = {eg:.3f} ± {eg_err:.3f} eV,  ln c = {p[0]:.2f} ± {np.sqrt(cov[0, 0]):.2f} "
          f"(c = {np.exp(p[0]):.2e} A K^-3),  chi2/dof = {chi2_dof:.1f}")

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    i0, i0_err = np.exp(ln_i0), np.exp(ln_i0) * ln_i0_err
    ax.errorbar(T, i0, yerr=i0_err, fmt="o", ms=5, color=BLUE, ecolor=BLUE, capsize=3,
                zorder=3, label=f"I₀ from eq. (4) fit at the {NID} minimum (~1–2 mA)")
    t_line = np.linspace(T.min() - 3, T.max() + 3, 300)
    ax.plot(t_line, np.exp(model_ln_i0(t_line, p, 3)), color=INK, lw=1.5, zorder=4,
            label=r"Fit: eq. (3), I₀ = c T³ exp(−$E_g$ / $k_B T$)")
    ax.set_yscale("log")
    lines = [f"$E_g$ = {eg:.3f} ± {eg_err:.3f} eV",
             f"c = {np.exp(p[0]):.1e} A K⁻³  (ln c = {p[0]:.1f} ± {np.sqrt(cov[0, 0]):.1f})",
             f"χ²/dof = {chi2_dof:.1f} (errors scaled by √(χ²/dof))"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title("GaAs LED saturation current I₀ against temperature", color=INK)
    ax.set_ylabel("I₀ (A, log scale)", color=MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style(ax)
    ax.grid(True, which="minor", color=GRID, linewidth=0.5, zorder=0)

    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.errorbar(T, ln_i0 - model_ln_i0(T, p, 3), yerr=ln_i0_err, fmt="o", ms=4,
                    color=BLUE, ecolor=FAINT, capsize=2, zorder=3)
    ax_res.set_xlabel("Temperature (K)", color=MUTED)
    ax_res.set_ylabel("ln I₀ − fit", color=MUTED)
    style(ax_res)

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(f"Plot -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
