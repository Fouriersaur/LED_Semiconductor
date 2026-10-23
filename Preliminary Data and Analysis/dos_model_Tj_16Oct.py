"""Junction temperature from the spectral shape with the two-sigmoid DOS model, 16 Oct.

Follows Vaskuri et al., J. Appl. Phys. 118, 203103 (2015), Sec. IV. The DOS is calibrated
on the averaged 293.5 K, 17 mA reference spectrum (dos_sigmoid_16Oct.py), and A1/A2, σ1, σ2
and ΔE2 are then
held fixed while every 16 Oct spectrum is fitted with Eq. (6),

    r(E) = S [ A1 / (1 + exp(-(E - Eg) / (σ1 + Δσ1)))
             + A2 / (1 + exp(-(E - Eg - ΔE2) / (σ2 + Δσ2))) ] exp(-(E - Eg) / k T_j)

with T_j, Eg, Δσ1 and Δσ2 free. S is a free scale: the paper normalises the spectra to their
peak, so the model describes the shape only. As for the DOS, the spectra are averaged into
1 nm bins and ln r(E) is fitted by least squares over the bins with S/N ≥ 10.

The fitted T_j is compared with the junction temperature from the pulse measurements, the
stage temperature plus the ΔT of dos_sigmoid_16Oct.junction_temperature.

As a check the model is also fitted to the reference spectrum itself, which should return
its own junction temperature. Above ~2.02 eV the reference DOS falls (its high-energy tail is steeper
than a 293.5 K Boltzmann factor), which the rising sigmoids cannot follow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import least_squares

from IV_Curve_25C import BLUE, FAINT, GRID, INK, K_B_OVER_Q, MUTED, ORANGE, style
from spectrum_bandgap import TEMP_CMAP, parse_temperature
from dos_sigmoid_16Oct import (BACKGROUND, CURRENT_MA, OUTPUT_DIR, SPECTRA_DIR, binned_spectrum,
                               calibration, junction_temperature, signal_window, sigmoid)

PARAM_NAMES = ("lnS", "Eg", "Tj", "dsigma1", "dsigma2")
# Spectra shown together in the Fig. 4-style plot (nearest measured temperature to each)
OVERLAY_T_K = (76.7, 104, 145, 190, 230, 268.5)
# Starting T_j as multiples of the junction temperature from the pulses: the fit has local minima
START_TJ_FACTORS = (0.8, 1.0, 1.3)


def log_model(energy, ln_s, eg, tj, ds1, ds2, cal):
    """ln r(E) from Eq. (6) with the calibrated DOS parameters cal = (ln A1/A2, ΔE2, σ1, σ2)."""
    ln_ratio, de2, s1, s2 = cal
    dos = sigmoid((energy - eg) / (s1 + ds1)) \
        + np.exp(-ln_ratio) * sigmoid((energy - eg - de2) / (s2 + ds2))
    return ln_s + np.log(dos) - (energy - eg) / (K_B_OVER_Q * tj)


def fit_spectrum(energy, intensity, T_start, eg_start, cal):
    """Fit ln r(E); returns the parameters, their errors and the rms of ln r."""
    log_i = np.log(intensity)
    s1, s2 = cal[2], cal[3]

    def resid(p):
        return log_model(energy, *p, cal) - log_i

    lower = [-50, energy.min(), 10, -s1 + 2e-4, -s2 + 2e-4]
    upper = [50, energy.max(), 1000, 0.1, 0.1]
    starts = [[log_i.max(), eg_start, f * T_start, 0.0, 0.0] for f in START_TJ_FACTORS]
    best = min((least_squares(resid, p0, bounds=(lower, upper), x_scale=[1, 1e-3, 10, 1e-3, 1e-3])
                for p0 in starts), key=lambda r: r.cost)
    dof = len(energy) - len(best.x)
    rms = np.sqrt(2 * best.cost / dof)
    cov = np.linalg.pinv(best.jac.T @ best.jac) * rms**2
    return best.x, np.sqrt(np.diag(cov)), rms


def fit_all(cal, eg_cal, e_peak_cal):
    files = sorted(SPECTRA_DIR.glob("161026_*.csv"), key=lambda f: parse_temperature(f)[0])
    results = []
    for path in files:
        T, dT = parse_temperature(path)
        Tc, Tc_err = junction_temperature(T, dT)
        e, intensity, noise = binned_spectrum(path, BACKGROUND)
        used = signal_window(intensity, noise)
        e_peak = e[np.argmax(intensity)]
        # Eg follows the peak: start from the calibrated Eg shifted with it
        p, err, rms = fit_spectrum(e[used], intensity[used], Tc, eg_cal + e_peak - e_peak_cal, cal)
        results.append(dict(T=T, dT=dT, Tc=Tc, Tc_err=Tc_err, e=e[used], I=intensity[used],
                            E_peak=e_peak, p=p, err=err, rms=rms, name=path.stem))
    return results


def model_curve(e, p, cal):
    return np.exp(log_model(e, *p, cal))


def plot_spectra(results, cal, cal_tj, path):
    """Like Fig. 4 of the paper: chosen spectra with the model, log and linear, and differences."""
    chosen = [min(results, key=lambda r: abs(r["T"] - t)) for t in OVERLAY_T_K]
    norm = Normalize(min(r["T"] for r in chosen), max(r["T"] for r in chosen))
    fig, (ax_log, ax_lin, ax_diff) = plt.subplots(3, 1, figsize=(8, 11), sharex=True,
                                                  gridspec_kw={"height_ratios": [3, 3, 2]})
    for r in chosen:
        color = TEMP_CMAP(0.2 + 0.8 * norm(r["T"]))
        peak = r["I"].max()
        e_line = np.linspace(r["e"].min(), r["e"].max(), 600)
        label = f"{r['p'][2]:.1f} K ({r['Tc']:.1f} K)"
        for ax in (ax_log, ax_lin):
            ax.plot(r["e"], r["I"] / peak, "o", ms=3, mfc="white", color=color, zorder=3)
            ax.plot(e_line, model_curve(e_line, r["p"], cal) / peak, color=color, lw=1.3,
                    zorder=4, label=label)
        model = model_curve(r["e"], r["p"], cal)
        ax_diff.plot(r["e"], (r["I"] - model) / model * 100, "o-", ms=2.5, lw=1, color=color)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("r(E) / peak (log scale)", color=MUTED)
    ax_log.set_title("(a) Log scale", color=INK)
    ax_log.legend(title="Fitted $T_j$ ($T_j$ from pulses)", loc="upper left", fontsize=8,
                  title_fontsize=8, frameon=False)
    ax_lin.set_ylabel("r(E) / peak", color=MUTED)
    ax_lin.set_title("(b) Linear scale", color=INK)
    ax_diff.axhline(0, color=MUTED, lw=1)
    ax_diff.set_ylabel("(data − model) / model (%)", color=MUTED)
    ax_diff.set_title("(c) Relative difference", color=INK)
    ax_diff.set_xlabel("Photon energy E (eV)", color=MUTED)
    for ax in (ax_log, ax_lin, ax_diff):
        style(ax)
    fig.suptitle(f"Two-sigmoid DOS model, Eq. (6), on the 16 Oct spectra ({CURRENT_MA} mA)\n"
                 f"DOS calibrated at $T_j$ = {cal_tj:.1f} K; $T_j$, $E_g$, Δσ1, Δσ2 fitted",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_single(r, cal, path):
    """One spectrum with its fit on a log scale, and the relative difference below."""
    fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    peak = r["I"].max()
    e_line = np.linspace(r["e"].min(), r["e"].max(), 600)
    p, err = r["p"], r["err"]
    ax.plot(r["e"], r["I"] / peak, "o", ms=4, mfc="white", color=BLUE, zorder=3, label="Measured")
    ax.plot(e_line, model_curve(e_line, p, cal) / peak, color=INK, lw=1.5, zorder=4,
            label="Model, Eq. (6)")
    ax.axvline(p[1], color=MUTED, lw=1, ls="--", zorder=2)
    ax.set_yscale("log")
    ax.set_ylabel("r(E) / peak (log scale)", color=MUTED)
    ax.set_title(f"LED spectrum at {r['T']:.1f} ± {r['dT']:.1f} K stage, $T_j$ = "
                 f"{r['Tc']:.1f} ± {r['Tc_err']:.1f} K from pulses, {CURRENT_MA} mA",
                 color=INK, fontsize=10)
    lines = [f"$T_j$ = {p[2]:.1f} ± {err[2]:.1f} K",
             f"$E_g$ = {p[1]:.4f} ± {err[1]:.4f} eV",
             f"Δσ1 = {p[3] * 1e3:.2f} ± {err[3] * 1e3:.2f} meV",
             f"Δσ2 = {p[4] * 1e3:.2f} ± {err[4] * 1e3:.2f} meV",
             f"rms(ln r) = {r['rms']:.3f}"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    model = model_curve(r["e"], p, cal)
    ax_diff.axhline(0, color=MUTED, lw=1)
    ax_diff.plot(r["e"], (r["I"] - model) / model * 100, "o", ms=3, color=BLUE)
    ax_diff.set_ylabel("Rel. diff. (%)", color=MUTED)
    ax_diff.set_xlabel("Photon energy E (eV)", color=MUTED)
    style(ax_diff)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tj(results, cal_tj, path):
    """Fitted T_j against the junction temperature from the pulses, and their difference."""
    T, dT = np.array([r["Tc"] for r in results]), np.array([r["Tc_err"] for r in results])
    tj, tj_err = np.array([r["p"][2] for r in results]), np.array([r["err"][2] for r in results])
    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    lim = (min(T.min(), tj.min()) - 5, max(T.max(), tj.max()) + 5)
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=2, label="Fitted = from pulses")
    ax.errorbar(T, tj, xerr=dT, yerr=tj_err, fmt="o", ms=4, color=BLUE, ecolor=FAINT,
                capsize=2, zorder=3, label="Fitted $T_j$")
    ax.set_ylabel("Fitted $T_j$ (K)", color=MUTED)
    ax.set_title(f"Junction temperature from the DOS model (calibrated at {cal_tj:.1f} K)",
                 color=INK)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax)
    ax_d.axhline(0, color=MUTED, lw=1)
    ax_d.errorbar(T, tj - T, xerr=dT, yerr=tj_err, fmt="o", ms=4, color=BLUE, ecolor=FAINT,
                  capsize=2, zorder=3)
    ax_d.set_xlabel("$T_j$ from pulses: stage T + ΔT (K)", color=MUTED)
    ax_d.set_ylabel("Fitted − from pulses (K)", color=MUTED)
    style(ax_d)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_parameters(results, path):
    """Like Fig. 5 of the paper: Eg, E_peak, Δσ1 and Δσ2 against the fitted T_j."""
    tj = np.array([r["p"][2] for r in results])
    eg, eg_err = np.array([r["p"][1] for r in results]), np.array([r["err"][1] for r in results])
    e_peak = np.array([r["E_peak"] for r in results])
    ds = np.array([r["p"][3:5] for r in results]) * 1e3
    ds_err = np.array([r["err"][3:5] for r in results]) * 1e3
    fig, (ax_e, ax_s) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_e.errorbar(tj, eg, yerr=eg_err, fmt="o", ms=4, color=BLUE, ecolor=FAINT, capsize=0,
                  zorder=3, label="$E_g$ (fitted)")
    ax_e.plot(tj, e_peak, "s", ms=4, mfc="white", color=ORANGE, zorder=3,
              label="$E_{peak}$ (measured)")
    ax_e.set_ylabel("Energy (eV)", color=MUTED)
    ax_e.set_title("Fitted parameters against the fitted junction temperature", color=INK)
    ax_e.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax_e)
    ax_s.errorbar(tj, ds[:, 0], yerr=ds_err[:, 0], fmt="o", ms=4, color=BLUE, ecolor=FAINT,
                  capsize=0, zorder=3, label="Δσ1")
    ax_s.errorbar(tj, ds[:, 1], yerr=ds_err[:, 1], fmt="s", ms=4, mfc="white", color=INK,
                  ecolor=FAINT, capsize=0, zorder=3, label="Δσ2")
    ax_s.axhline(0, color=MUTED, lw=1)
    ax_s.set_xlabel("Fitted $T_j$ (K)", color=MUTED)
    ax_s.set_ylabel("Broadening Δσ (meV)", color=MUTED)
    ax_s.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax_s)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def temperature_coefficient(x, tj):
    """Slope of a straight-line fit of x against T_j, as Eq. (7) of the paper."""
    return np.polyfit(tj, x, 1)[0]


def main():
    c = calibration()
    p_cal = c["fit"][0]
    # (ln A1/A2, ΔE2, σ1, σ2) from the reference DOS fit
    cal = (p_cal[3], p_cal[4], p_cal[2], p_cal[5])
    print(f"Calibration (T_j = {c['T']:.1f} K): A1/A2 = {np.exp(cal[0]):.3g}, ΔE2 = {cal[1] * 1e3:.1f} meV, "
          f"σ1 = {cal[2] * 1e3:.2f} meV, σ2 = {cal[3] * 1e3:.2f} meV")

    used = c["used"]
    p, err, rms = fit_spectrum(c["e"][used], c["intensity"][used], c["T"], p_cal[1], cal)
    print(f"Reference fitted with its own calibration: Tj = {p[2]:.1f} ± {err[2]:.1f} K "
          f"(should be {c['T']:.1f} K), rms(ln r) = {rms:.3f}")

    results = fit_all(cal, p_cal[1], c["e_peak"])
    print("\n  stage T (K)   Tj pulses (K)   Tj fitted (K)  Eg (eV)    Δσ1 (meV)  Δσ2 (meV)  "
          "rms(ln r)")
    for r in results:
        p, err = r["p"], r["err"]
        print(f"  {r['T']:6.1f} ± {r['dT']:.1f}  {r['Tc']:6.1f} ± {r['Tc_err']:.1f}   "
              f"{p[2]:6.1f} ± {err[2]:4.1f}  {p[1]:.4f}   "
              f"{p[3] * 1e3:7.2f}    {p[4] * 1e3:7.2f}    {r['rms']:.3f}")
    tj = np.array([r["p"][2] for r in results])
    T = np.array([r["Tc"] for r in results])
    print(f"\nMean |Tj fitted - Tj pulses| = {np.mean(np.abs(tj - T)):.1f} K, "
          f"max = {np.max(np.abs(tj - T)):.1f} K")
    print("Temperature coefficients against the fitted Tj (as Table II of the paper):")
    print(f"  Eg: {temperature_coefficient([r['p'][1] for r in results], tj) * 1e6:.0f} µeV/K,"
          f"  Δσ1: {temperature_coefficient([r['p'][3] for r in results], tj) * 1e6:.1f} µeV/K,"
          f"  Δσ2: {temperature_coefficient([r['p'][4] for r in results], tj) * 1e6:.1f} µeV/K")

    single_dir = OUTPUT_DIR / "16_Oct_DOS_model_fits"
    single_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        plot_single(r, cal, single_dir / f"16_Oct_DOS_model_{r['T']:.1f}K.png")
    plot_spectra(results, cal, c["T"], OUTPUT_DIR / "16_Oct_DOS_model_spectra.png")
    plot_tj(results, c["T"], OUTPUT_DIR / "16_Oct_DOS_model_Tj_vs_T.png")
    plot_parameters(results, OUTPUT_DIR / "16_Oct_DOS_model_parameters_vs_Tj.png")
    print(f"\nPlots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
