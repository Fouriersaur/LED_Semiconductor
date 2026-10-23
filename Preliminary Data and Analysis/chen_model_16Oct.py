"""Junction temperature of the 16 Oct spectra with the model of Chen et al. (2009).

N. C. Chen et al., "Spectral shape and broadening of emission from AlGaInP light-emitting
diodes", J. Appl. Phys. 106, 074514 (2009). For a quantum well the joint DOS is a step at
the effective band gap Eg, and Eg itself is spread over a distribution W(Eg), so the photon
density spectrum is (their Eq. 4)

    r(hν) ∝ exp(-hν / kT) ∫_0^hν W(Eg) dEg.

Following the paper:

1. The spectrum is turned into a photon density, r(hν) = P(hν) / hν, where P(hν) is the
   intensity per eV of load_spectrum (binned into 1 nm bins as in dos_sigmoid_16Oct.py).
2. T_m from the gradient of the high-energy tail, T = -(1/k) [∂ ln r / ∂hν]^-1 (Eq. 10),
   fitted over the bins between TAIL_FRACTION of the peak.
3. r exp(hν / kT_m), normalised to one at high energy, is the integral of W(Eg) (Eq. 6), and
   its derivative is W(Eg) (Eq. 7). W is close to a Gaussian of centre E0 and FWHM Δ (Eq. 8),
   so the integral is fitted with a Gaussian CDF, which is steadier than differentiating.
4. T_m is corrected for the broadening with Eq. 11, T_m = T_c / (1 - G/∫G kT_c), at the
   middle of the tail window.
5. The whole spectrum is fitted with the analytic Eq. 9,

       r(hν) = C exp(-hν / kT) · ½ erfc(-2√ln2 (hν - E0) / Δ),

   with C, T, E0 and Δ free, on a linear scale as in Fig. 7(a), over the bins above
   EQ9_FRACTION of the peak. In the paper T_j is the tail gradient (Fig. 6) and Eq. 9 only
   checks the shape with that T, E0 and Δ; leaving T free here tests whether the two agree.
   A fit of ln r instead is pulled off by the low-energy wing, which at low temperature
   falls exponentially rather than as a Gaussian.

Errors are statistical (scaled by the reduced χ²) and half the spread over the alternative
windows. All results are compared with the stage temperature plus the ΔT from the pulse
measurements (dos_sigmoid_16Oct.junction_temperature).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc
from scipy.stats import norm

from IV_Curve_25C import BLUE, FAINT, GRID, INK, K_B_OVER_Q, MUTED, ORANGE, PLOTS_DIR, style
from spectrum_bandgap import parse_temperature
from dos_sigmoid_16Oct import (BACKGROUND, CURRENT_MA, SIGNAL_TO_NOISE_MIN, SPECTRA_DIR,
                               binned_spectrum, junction_temperature)
from high_energy_tail_16Oct import fit_tail, tail_window

OUTPUT_DIR = PLOTS_DIR / "spectra" / "16_Oct_Chen_model"
# Tail window for Eq. 10, as fractions of the peak (upper, lower), and alternatives
TAIL_FRACTION = (0.3, 0.01)
ALT_TAIL_FRACTIONS = ((0.5, 0.01), (0.3, 0.03), (0.1, 0.005), (0.5, 0.05))
# Eq. 9 is fitted to the bins above this fraction of the peak, and alternatives
EQ9_FRACTION = 0.01
ALT_EQ9_FRACTIONS = (0.003, 0.03, 0.1)
# Fraction of the peak above which r exp(hν/kT) is used for W(Eg)
W_FRACTION = 0.003
SHOWN_T_K = 76.7
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def photon_density(path):
    """Energy, photon density r(hν) normalised to its peak, and its noise (1 nm bins)."""
    e, intensity, noise = binned_spectrum(path, BACKGROUND)
    r, r_noise = intensity / e, noise / e
    scale = r.max()
    return e, r / scale, r_noise / scale


def above(r, noise, fraction):
    """Mask of the contiguous bins around the peak above fraction of it and S/N ≥ the minimum."""
    peak = np.argmax(r)
    ok = (r >= fraction) & (r >= SIGNAL_TO_NOISE_MIN * noise)
    lo, hi = peak, peak
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(r) - 1 and ok[hi + 1]:
        hi += 1
    used = np.zeros_like(r, dtype=bool)
    used[lo:hi + 1] = True
    return used


def gaussian_cdf(e, e0, delta):
    """∫ W(Eg) dEg up to e for a unit-area Gaussian of centre e0 and FWHM delta (Eq. 8)."""
    return 0.5 * erfc(-2 * np.sqrt(np.log(2)) * (e - e0) / delta)


def band_gap_distribution(e, r, noise, tm, used):
    """Eqs. 6-8: r exp(hν/kT_m) fitted with A × Gaussian CDF; returns (A, E0, Δ) and the data."""
    y = r * np.exp((e - e[np.argmax(r)]) / (K_B_OVER_Q * tm))
    y_err = noise * np.exp((e - e[np.argmax(r)]) / (K_B_OVER_Q * tm))
    x, y, y_err = e[used], y[used], y_err[used]

    def resid(p):
        return (p[0] * gaussian_cdf(x, p[1], p[2]) - y) / y_err

    p0 = [y[-5:].mean(), x[np.argmax(np.gradient(y, x))], 0.02]
    fit = least_squares(resid, p0, bounds=([0, x.min(), 1e-3], [np.inf, x.max(), 0.2]))
    return fit.x, x, y / fit.x[0], y_err / fit.x[0]


def corrected_temperature(tm, e_mid, e0, delta):
    """Solve Eq. 11, T_m = T_c / (1 - G(hν - E0)/∫G · kT_c), for T_c at hν = e_mid."""
    sigma = delta / FWHM_PER_SIGMA
    ratio = norm.pdf(e_mid, e0, sigma) / norm.cdf(e_mid, e0, sigma)  # G / ∫G, per eV
    tc = tm
    for _ in range(50):
        tc = tm * (1 - ratio * K_B_OVER_Q * tc)
    return tc


def log_eq9(e, ln_c, tj, e0, delta):
    """ln r(hν) of Eq. 9, with the Boltzmann factor taken relative to e0 to keep ln_c small."""
    return ln_c - (e - e0) / (K_B_OVER_Q * tj) + np.log(gaussian_cdf(e, e0, delta))


def fit_eq9(e, r, noise, used, start):
    """Least squares of r with Eq. 9 on a linear scale, as Fig. 7(a) of the paper, weighted
    by the noise; returns params, errors and reduced χ².

    On a linear scale the fit follows the peak, where the Gaussian W(Eg) holds, and not the
    low-intensity wings, where it does not (the paper notes the same for W in Fig. 7(b)).
    """
    x, y, sigma = e[used], r[used], noise[used]

    def resid(p):
        return (np.exp(log_eq9(x, *p)) - y) / sigma

    lower, upper = [-20, 10, x.min() - 0.1, 1e-3], [20, 1000, x.max() + 0.1, 0.2]
    start = np.clip(start, np.add(lower, 1e-6), np.subtract(upper, 1e-6))
    fit = least_squares(resid, start, x_scale=[1, 10, 1e-3, 1e-3], bounds=(lower, upper))
    chi2 = 2 * fit.cost / (len(x) - len(fit.x))
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * max(chi2, 1.0)
    return fit.x, np.sqrt(np.diag(cov)), chi2


def analyse(path):
    T, dT = parse_temperature(path)
    tc_pulses, tc_err = junction_temperature(T, dT)
    e, r, noise = photon_density(path)
    out = dict(T=T, dT=dT, Tp=tc_pulses, Tp_err=tc_err, e=e, r=r, noise=noise)

    # Eq. 10: gradient of the high-energy tail
    tail = tail_window(r, noise, TAIL_FRACTION)
    tm, tm_err, line = fit_tail(e, r, noise, tail)
    alt = [fit_tail(e, r, noise, tail_window(r, noise, f))[0] for f in ALT_TAIL_FRACTIONS]
    tm_sys = (max(alt + [tm]) - min(alt + [tm])) / 2
    out.update(Tm=tm, Tm_err=np.hypot(tm_err, tm_sys), tail=tail, line=line)

    # Eqs. 6-8: band gap distribution, and Eq. 11: T_m corrected for the broadening
    (a, e0, delta), wx, wy, wy_err = band_gap_distribution(e, r, noise, tm,
                                                           above(r, noise, W_FRACTION))
    e_mid = e[tail].mean()
    out.update(W=(wx, wy, wy_err), E0_W=e0, delta_W=delta,
               Tc11=corrected_temperature(tm, e_mid, e0, delta), e_mid=e_mid)

    # Eq. 9: the whole spectrum
    used = above(r, noise, EQ9_FRACTION)
    start = [np.log(r.max()), tm, e0, delta]
    p, err, chi2 = fit_eq9(e, r, noise, used, start)
    alt = [fit_eq9(e, r, noise, above(r, noise, f), p)[0][1] for f in ALT_EQ9_FRACTIONS]
    sys = (max(alt + [p[1]]) - min(alt + [p[1]])) / 2
    out.update(eq9=p, eq9_err=err, eq9_sys=sys, Tj=p[1], Tj_err=np.hypot(err[1], sys),
               chi2=chi2, used=used)
    return out


def plot_spectrum(s, path):
    """Like Figs. 3 and 7 of the paper: r(hν) with Eq. 9 and the tail line, and W(Eg)."""
    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 1, 0.08], width_ratios=[1.15, 1])
    ax_log = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[1, 0], sharex=ax_log)
    ax_lin = fig.add_subplot(grid[0, 1])
    ax_w = fig.add_subplot(grid[1, 1])

    e, r, noise, used = s["e"], s["r"], s["noise"], s["used"]
    p = s["eq9"]
    peak = e[np.argmax(r)]
    e_line = np.linspace(peak - 0.08, peak + 0.08, 800)
    model = np.exp(log_eq9(e_line, *p))
    c, b = s["line"]

    ax_log.errorbar(e[~used], r[~used], yerr=noise[~used], fmt="o", ms=3, mfc="white",
                    color=FAINT, ecolor=FAINT, zorder=2, label="Measured, not in Eq. 9 fit")
    ax_log.errorbar(e[used], r[used], yerr=noise[used], fmt="o", ms=3.5, color=BLUE,
                    ecolor=BLUE, zorder=3, label="Measured, in Eq. 9 fit")
    ax_log.plot(e_line, model, color=INK, lw=1.6, zorder=4,
                label=f"Eq. 9: $T_j$ = {s['Tj']:.1f} ± {s['Tj_err']:.1f} K")
    tail_e = np.linspace(peak, e[s["tail"]].max() + 0.03, 100)
    ax_log.plot(tail_e, np.exp(c + b * tail_e), color=ORANGE, lw=1.4, ls="--", zorder=4,
                label=f"Tail gradient, Eq. 10: $T_m$ = {s['Tm']:.1f} ± {s['Tm_err']:.1f} K")
    ax_log.set_yscale("log")
    ax_log.set_ylim(max(noise.min(), 1e-4), 2)
    ax_log.set_xlim(peak - 0.08, peak + 0.08)
    ax_log.set_ylabel("Photon density r(hν) / peak (log scale)", color=MUTED)
    ax_log.set_title("(a) Photon density spectrum and fits", color=INK)
    ax_log.legend(loc="lower center", fontsize=8.5, frameon=False)
    style(ax_log)

    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.errorbar(e[used], np.log(r[used]) - log_eq9(e[used], *p),
                    yerr=noise[used] / r[used], fmt="o", ms=3, color=BLUE, ecolor=FAINT)
    ax_res.set_ylabel("ln r − Eq. 9", color=MUTED)
    ax_res.set_xlabel("Photon energy hν (eV)", color=MUTED)
    style(ax_res)

    ax_lin.plot(e, r, "o", ms=3.5, mfc="white", color=BLUE, zorder=3, label="Measured")
    ax_lin.plot(e_line, model, color=INK, lw=1.6, zorder=4, label="Eq. 9")
    ax_lin.set_xlim(peak - 0.06, peak + 0.06)
    ax_lin.set_ylabel("r(hν) / peak", color=MUTED)
    ax_lin.set_title("(b) Linear scale, as Fig. 7(a)", color=INK)
    lines = [f"$T_j$ = {p[1]:.1f} ± {s['eq9_err'][1]:.1f} (stat) ± {s['eq9_sys']:.1f} K",
             f"$E_0$ = {p[2]:.4f} ± {s['eq9_err'][2]:.4f} eV",
             f"Δ = {p[3] * 1e3:.1f} ± {s['eq9_err'][3] * 1e3:.1f} meV",
             f"reduced χ² = {s['chi2']:.1f}",
             f"$T_j$ from pulses = {s['Tp']:.1f} ± {s['Tp_err']:.1f} K"]
    ax_lin.text(0.03, 0.97, "\n".join(lines), transform=ax_lin.transAxes, va="top",
                fontsize=8.5, color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax_lin.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax_lin)

    # W(Eg) as the derivative of r exp(hν/kT_m) (Eq. 7), with the fitted Gaussian
    wx, wy, _ = s["W"]
    w = np.gradient(wy, wx)
    sigma = s["delta_W"] / FWHM_PER_SIGMA
    x_line = np.linspace(wx.min(), wx.max(), 400)
    ax_w.plot(wx, w, "o", ms=3.5, color=BLUE, zorder=3, label="d[r e$^{hν/kT_m}$]/dhν")
    ax_w.plot(x_line, norm.pdf(x_line, s["E0_W"], sigma), color=INK, lw=1.4, zorder=4,
              label=f"Gaussian: $E_0$ = {s['E0_W']:.4f} eV, Δ = {s['delta_W'] * 1e3:.1f} meV")
    ax_w.axhline(0, color=MUTED, lw=1)
    ax_w.set_xlim(peak - 0.06, peak + 0.06)
    ax_w.set_ylabel("W($E_g$) (1/eV)", color=MUTED)
    ax_w.set_xlabel("$E_g$ (eV)", color=MUTED)
    ax_w.set_title("(c) Band gap distribution, Eq. 7, as Fig. 7(b)", color=INK)
    ax_w.legend(loc="upper left", fontsize=8, frameon=False)
    style(ax_w)

    fig.suptitle(f"Chen et al. (2009) model, {s['T']:.1f} ± {s['dT']:.1f} K stage, "
                 f"{CURRENT_MA} mA", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(results, path, dt_label="ΔT linear in T"):
    """T_j from Eq. 9, and T_m from Eq. 10 before and after Eq. 11, against the pulses.

    dt_label names the interpolation of the pulse ΔT used for the x axis.
    """
    tp = np.array([s["Tp"] for s in results])
    tp_err = np.array([s["Tp_err"] for s in results])
    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    lim = (tp.min() - 5, tp.max() + 5)
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=2, label="= from pulses")
    series = (("Tj", "Tj_err", "Eq. 9 fit", "o", BLUE, BLUE),
              ("Tm", "Tm_err", "Tail gradient, Eq. 10", "s", ORANGE, "white"),
              ("Tc11", "Tm_err", "Eq. 10 corrected with Eq. 11", "^", INK, "white"))
    for key, err_key, label, fmt, color, face in series:
        t = np.array([s[key] for s in results])
        err = np.array([s[err_key] for s in results])
        for a, y in ((ax, t), (ax_d, t - tp)):
            a.errorbar(tp, y, xerr=tp_err, yerr=err, fmt=fmt, ms=4, mfc=face, color=color,
                       ecolor=FAINT, capsize=2, zorder=3, label=label)
    ax.set_ylabel("$T_j$ from the spectrum (K)", color=MUTED)
    ax.set_title(f"Junction temperature with the Chen et al. (2009) model, 16 Oct "
                 f"({CURRENT_MA} mA)", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax)
    ax_d.axhline(0, color=MUTED, lw=1)
    ax_d.set_xlabel(f"$T_j$ from pulses: stage T + ΔT (K), {dt_label}", color=MUTED)
    ax_d.set_ylabel("Spectrum − pulses (K)", color=MUTED)
    style(ax_d)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    files = sorted(SPECTRA_DIR.glob("161026_*.csv"), key=lambda f: parse_temperature(f)[0])
    results = [analyse(f) for f in files]

    print("  stage T (K)   Tj pulses (K)   Tm Eq.10 (K)    Tc Eq.11 (K)  "
          "Tj Eq.9 (K)              E0 (eV)   Δ (meV)  χ²")
    for s in results:
        p, err = s["eq9"], s["eq9_err"]
        print(f"  {s['T']:6.1f} ± {s['dT']:.1f}  {s['Tp']:6.1f} ± {s['Tp_err']:.1f}   "
              f"{s['Tm']:6.1f} ± {s['Tm_err']:4.1f}   {s['Tc11']:6.1f}       "
              f"{s['Tj']:6.1f} ± {err[1]:.1f} ± {s['eq9_sys']:4.1f}    {p[2]:.4f}   "
              f"{p[3] * 1e3:5.1f}   {s['chi2']:5.1f}")
    tp = np.array([s["Tp"] for s in results])
    for key, name in (("Tm", "Eq. 10"), ("Tc11", "Eq. 10 + 11"), ("Tj", "Eq. 9")):
        d = np.array([s[key] for s in results]) - tp
        print(f"{name}: mean (spectrum - pulses) = {d.mean():+.1f} K, "
              f"rms = {np.sqrt(np.mean(d**2)):.1f} K")

    fits_dir = OUTPUT_DIR / "16_Oct_Chen_model_fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for s in results:
        plot_spectrum(s, fits_dir / f"16_Oct_Chen_model_{s['T']:.1f}K.png")
    shown = min(results, key=lambda s: abs(s["T"] - SHOWN_T_K))
    plot_spectrum(shown, OUTPUT_DIR / f"16_Oct_Chen_model_{shown['T']:.1f}K.png")
    plot_summary(results, OUTPUT_DIR / "16_Oct_Chen_model_Tj_vs_T.png")
    print(f"\nPlots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
