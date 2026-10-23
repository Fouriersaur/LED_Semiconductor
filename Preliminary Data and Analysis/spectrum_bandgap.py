"""Extract the LED bandgap from its emission spectrum at each temperature.

Reads the Thorlabs CCS200 exports in data/LED_spectra_15Oct, named real_<T>_pm<dT>.csv for
T ± dT in kelvin with "_" as the decimal point (real_84_0_pm0_2 = 84.0 ± 0.2 K).

Each spectrum is converted to intensity per unit energy, I(E) = I(λ) λ² / (hc), and fitted
with band-to-band emission, a joint density of states times a Boltzmann tail,

    I(E) = A [ sqrt(E - Eg) exp(-(E - Eg) / kT_c) ] ⊗ sech(E / γ)

where T_c is the carrier temperature and the sech kernel (exponential tails on both
sides) models the broadening from alloy disorder and the Urbach tail below the gap.
Eg is then plotted against the measured temperature and fitted with a straight line.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.optimize import curve_fit

from IV_Curve_25C import BLUE, FAINT, ORANGE, GRID, INK, K_B_OVER_Q, MUTED, PLOTS_DIR, style

SPECTRA_DIR = Path(__file__).with_name("data") / "LED_spectra_15Oct"
PLOT_PREFIX = "15_Oct_spectrum_LED"

HC_EV_NM = 1239.84193
# Energies (eV) well away from the emission line, where the signal is the dark offset
BASELINE_OUTSIDE_EV = (1.8, 2.3)
# Fit window (eV), and alternative windows used for the systematic error on Eg
FIT_RANGE_EV = (1.90, 2.20)
ALT_FIT_RANGES_EV = [(1.95, 2.15), (1.98, 2.12), (1.85, 2.25)]
PLOT_RANGE_EV = (1.85, 2.25)

# Model grid: the convolution is done on a uniform grid in E - Eg, then interpolated
GRID_STEP_EV = 2e-4
_X = np.arange(-0.4, 0.4 + GRID_STEP_EV / 2, GRID_STEP_EV)
_G = np.arange(-0.2, 0.2 + GRID_STEP_EV / 2, GRID_STEP_EV)

# Light (cold) to dark (hot): one-hue sequential ramp for the overlay
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "temperature", ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

PARAM_NAMES = ("A", "Eg", "kTc", "gamma")

# Varshni β (K): 18 K of data cannot fix it, so it is also fitted at fixed values.
# GaAs: β = 204 K (Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001))
VARSHNI_BETA_GAAS_K = 204
VARSHNI_BETAS_K = (100, VARSHNI_BETA_GAAS_K, 300, 400, 1000, np.inf)

# Theoretical Eg(T) of the Würth 151033RS03000 red AlGaInP LED: (Al_x Ga_1-x)_0.51 In_0.49 P,
# lattice-matched to GaAs, with x set so Eg(298 K) is the datasheet peak (628 ± 2 nm at 25 °C)
THEORY_GROUP_III_FRACTION = 0.51
THEORY_PEAK_NM, THEORY_PEAK_TOL_NM = 628.0, 2.0
N_AIR = 1.00027


def parse_temperature(path):
    """(T, dT) in K from a name like real_84_0_pm0_2 (also real_91_9pm0_2, 161026_199_3_pm0_2)."""
    m = re.fullmatch(r"[^_]+_(\d+)_(\d+)_?pm(\d+)_(\d+)", path.stem)
    if m is None:
        raise ValueError(f"cannot read a temperature from {path.name}")
    a, b, c, d = m.groups()
    return float(f"{a}.{b}"), float(f"{c}.{d}")


def read_spectrum(path):
    """Wavelength (nm) and raw intensity from the [Data] section of a CCS200 export."""
    rows, in_data = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[Data]":
                in_data = True
            elif line.startswith("["):
                in_data = False
            elif in_data and line and not line.startswith("#"):
                rows.append([float(x) for x in line.split(";")])
    return np.array(rows).T


def load_spectrum(path, background=None):
    """Energy (eV, ascending) and baseline-subtracted intensity per unit energy.

    background, if given, is a spectrometer background file on the same pixels, subtracted
    pixel by pixel before the remaining offset is removed. Its wavelengths may be in vacuum
    where the spectrum's are in air (or the other way round), which only rescales them by
    the refractive index of air.
    """
    wavelength, intensity = read_spectrum(path)
    if background is not None:
        bg_wavelength, bg_intensity = read_spectrum(background)
        ratio = bg_wavelength / wavelength if len(bg_wavelength) == len(wavelength) else None
        if ratio is None or not any(np.allclose(ratio, n, atol=5e-5)
                                    for n in (1, N_AIR, 1 / N_AIR)):
            raise ValueError(f"{background.name} is not on the wavelength grid of {path.name}")
        intensity = intensity - bg_intensity
    energy = HC_EV_NM / wavelength
    off_peak = (energy < BASELINE_OUTSIDE_EV[0]) | (energy > BASELINE_OUTSIDE_EV[1])
    intensity = intensity - np.median(intensity[off_peak])
    # Per nm -> per eV: |dλ/dE| = λ² / hc
    intensity_e = intensity * wavelength**2 / HC_EV_NM
    order = np.argsort(energy)
    return energy[order], intensity_e[order]


def model(energy, a, eg, kt_c, gamma):
    """sqrt(E - Eg) exp(-(E - Eg)/kT_c), convolved with sech(E / γ), times A."""
    x = np.clip(_X, 0, None)
    emission = np.where(_X > 0, np.sqrt(x) * np.exp(-x / kt_c), 0.0)
    kernel = 1 / np.cosh(_G / gamma)
    broadened = np.convolve(emission, kernel / kernel.sum(), mode="same")
    return a * np.interp(energy - eg, _X, broadened)


def fit_spectrum(energy, intensity, fit_range):
    """Least-squares fit in fit_range. Returns params, their errors and the residual rms."""
    m = (energy > fit_range[0]) & (energy < fit_range[1])
    e, i = energy[m], intensity[m]
    e_peak = e[np.argmax(i)]
    p0 = [i.max() / 0.04, e_peak - 0.005, 0.008, 0.005]
    p, cov = curve_fit(
        model, e, i, p0=p0, diff_step=1e-3, x_scale=[p0[0], 1e-3, 1e-3, 1e-3],
        bounds=([0, e_peak - 0.1, 1e-3, 2e-4], [np.inf, e_peak + 0.02, 0.05, 0.05]),
    )
    rms = np.std(i - model(e, *p))
    return dict(zip(PARAM_NAMES, p)), dict(zip(PARAM_NAMES, np.sqrt(np.diag(cov)))), rms


def plot_spectrum(energy, intensity, fit, T, dT, path, fit_range=FIT_RANGE_EV,
                  plot_range=PLOT_RANGE_EV):
    """Intensity against energy with the fit, and the residuals below."""
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    p = fit["p"]
    shown = (energy > plot_range[0]) & (energy < plot_range[1])
    in_fit = (energy > fit_range[0]) & (energy < fit_range[1])
    e_line = np.linspace(*fit_range, 1500)

    ax.scatter(energy[shown], intensity[shown], s=6, color=BLUE, zorder=3, label="Measured")
    ax.plot(e_line, model(e_line, *p.values()), color=INK, lw=1.5, zorder=4,
            label=r"Fit: $\sqrt{E - E_g}\,e^{-(E - E_g)/k T_c}$ ⊗ sech$(E/\gamma)$")
    ax.axvline(p["Eg"], color=MUTED, lw=1, ls="--", zorder=2)
    ax.annotate(f"$E_g$ = {p['Eg']:.4f} eV", (p["Eg"], 0.5 * intensity.max()),
                xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=9, color=INK)

    lines = [f"T = {T:.1f} ± {dT:.1f} K, fit {fit_range[0]:.3g}–{fit_range[1]:.3g} eV",
             f"$E_g$ = {p['Eg']:.4f} ± {fit['Eg_err']:.4f} eV",
             f"$T_c$ = {p['kTc'] / K_B_OVER_Q:.0f} K,  γ = {p['gamma'] * 1e3:.1f} meV",
             f"peak at {fit['E_peak']:.4f} eV"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title(f"LED emission spectrum at {T:.1f} K", color=INK)
    ax.set_ylabel("Intensity (arb. units per eV)", color=MUTED)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)

    resid = intensity - model(energy, *p.values())
    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.scatter(energy[shown & in_fit], resid[shown & in_fit], s=4, color=BLUE, zorder=3)
    ax_res.scatter(energy[shown & ~in_fit], resid[shown & ~in_fit], s=4, color=FAINT, zorder=3)
    ax_res.set_xlabel("Photon energy E (eV)", color=MUTED)
    ax_res.set_ylabel("Data − fit", color=MUTED)
    ax_res.set_xlim(*plot_range)
    style(ax_res)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spectrum_log(energy, intensity, fit, T, path, fit_range=FIT_RANGE_EV,
                      plot_range=PLOT_RANGE_EV):
    """Same spectrum on a log intensity scale, to show the exponential tails."""
    fig, ax = plt.subplots(figsize=(7, 5))
    p = fit["p"]
    shown = (energy > plot_range[0]) & (energy < plot_range[1]) & (intensity > 0)
    e_line = np.linspace(*fit_range, 1500)

    ax.scatter(energy[shown], intensity[shown], s=6, color=BLUE, zorder=3, label="Measured")
    ax.plot(e_line, model(e_line, *p.values()), color=INK, lw=1.5, zorder=4, label="Fit")
    ax.axvline(p["Eg"], color=MUTED, lw=1, ls="--", zorder=2, label=f"$E_g$ = {p['Eg']:.4f} eV")
    ax.set_yscale("log")
    ax.set_ylim(intensity.max() * 3e-4, intensity.max() * 2)
    ax.set_xlim(*plot_range)
    ax.set_title(f"LED emission spectrum at {T:.1f} K (log scale)", color=INK)
    ax.set_xlabel("Photon energy E (eV)", color=MUTED)
    ax.set_ylabel("Intensity (arb. units per eV, log scale)", color=MUTED)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_overlay(spectra, path, xlim_linear=(1.98, 2.10), xlim_log=(1.95, 2.13),
                 normalise=False):
    """All spectra on one plot, linear and log, coloured by temperature.

    With normalise, each spectrum is divided by its own peak, to compare shape and position.
    """
    temps = [s["T"] for s in spectra]
    norm = Normalize(min(temps), max(temps))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for s in spectra:
        shown = (s["E"] > xlim_log[0] - 0.05) & (s["E"] < xlim_log[1] + 0.05)
        color = TEMP_CMAP(norm(s["T"]))
        intensity = s["I"] / s["I"].max() if normalise else s["I"]
        for ax in axes:
            ax.plot(s["E"][shown], intensity[shown], color=color, lw=1)
    peak = 1.0 if normalise else max(s["I"].max() for s in spectra)
    unit = "Intensity / peak intensity" if normalise else "Intensity (arb. units per eV)"
    axes[0].set_xlim(*xlim_linear)
    axes[0].set_ylabel(unit, color=MUTED)
    axes[0].set_title("Linear scale", color=INK)
    axes[1].set_yscale("log")
    # Above the noise floor (~2e-3 of the peak), where 21 noisy traces would hide each other
    axes[1].set_ylim(peak * 1e-2, peak * 2)
    axes[1].set_xlim(*xlim_log)
    axes[1].set_ylabel(unit.replace(")", ", log scale)") if not normalise else f"{unit} (log scale)",
                       color=MUTED)
    axes[1].set_title("Log scale", color=INK)
    for ax in axes:
        ax.set_xlabel("Photon energy E (eV)", color=MUTED)
        style(ax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TEMP_CMAP), ax=axes, pad=0.02)
    cbar.set_label("Temperature (K)", color=MUTED)
    fig.suptitle(f"LED emission spectra, {min(temps):.1f}–{max(temps):.1f} K"
                 + (", normalised to peak" if normalise else ""), color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_line(T, T_err, eg, eg_err):
    """Weighted straight line Eg = Eg(T0) + slope (T - T0), T0 = mean T.

    Temperature errors enter through the effective variance σ² = σ_Eg² + (slope σ_T)².
    The covariance is scaled up by chi^2/dof when that exceeds 1.
    """
    t0 = T.mean()
    design = np.column_stack([np.ones_like(T), T - t0])
    slope = 0.0
    for _ in range(5):
        w = 1 / (eg_err**2 + (slope * T_err) ** 2)
        cov = np.linalg.inv(design.T @ (design * w[:, None]))
        p = cov @ design.T @ (w * eg)
        slope = p[1]
    chi2_dof = np.sum(w * (eg - design @ p) ** 2) / (len(T) - 2)
    return p, cov * max(chi2_dof, 1.0), chi2_dof, t0


def plot_eg_vs_t(T, T_err, eg, eg_err, path):
    p, cov, chi2_dof, t0 = fit_line(T, T_err, eg, eg_err)
    slope, slope_err = p[1], np.sqrt(cov[1, 1])
    print(f"\nEg(T) = Eg({t0:.1f} K) + dEg/dT (T - {t0:.1f} K):")
    print(f"  Eg({t0:.1f} K) = {p[0]:.5f} ± {np.sqrt(cov[0, 0]):.5f} eV,  "
          f"dEg/dT = {slope * 1e3:.3f} ± {slope_err * 1e3:.3f} meV/K,  chi2/dof = {chi2_dof:.1f}")

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.errorbar(T, eg, xerr=T_err, yerr=eg_err, fmt="o", ms=5, color=BLUE, ecolor=BLUE,
                capsize=3, zorder=3, label="$E_g$ from spectrum fit")
    t_line = np.linspace(T.min() - 1, T.max() + 1, 100)
    ax.plot(t_line, p[0] + slope * (t_line - t0), color=INK, lw=1.5, zorder=4,
            label="Fit: straight line")
    lines = [f"d$E_g$/dT = {slope * 1e3:.3f} ± {slope_err * 1e3:.3f} meV/K",
             f"$E_g$({t0:.1f} K) = {p[0]:.5f} ± {np.sqrt(cov[0, 0]):.5f} eV",
             f"χ²/dof = {chi2_dof:.1f}" + (" (errors scaled by √(χ²/dof))" if chi2_dof > 1 else "")]
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="right",
            fontsize=9, color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax.set_title("LED bandgap against temperature", color=INK)
    ax.set_ylabel("$E_g$ (eV)", color=MUTED)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    style(ax)

    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.errorbar(T, (eg - p[0] - slope * (T - t0)) * 1e3, xerr=T_err, yerr=eg_err * 1e3,
                    fmt="o", ms=4, color=BLUE, ecolor=FAINT, capsize=2, zorder=3)
    ax_res.set_xlabel("Temperature (K)", color=MUTED)
    ax_res.set_ylabel("$E_g$ − fit (meV)", color=MUTED)
    style(ax_res)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def varshni(T, e0, c, u):
    """Varshni Eg = E0 - α T² / (T + β), written with c = α/β and u = 1/β.

    u = 0 is the β → ∞ limit, Eg = E0 - c T², which the (α, β) form cannot reach.
    """
    return e0 - c * T**2 / (1 + u * T)


def varshni_slope(T, e0, c, u):
    return -c * T * (2 + u * T) / (1 + u * T) ** 2


def fit_varshni(T, T_err, eg, eg_err, u=None):
    """Weighted Varshni fit; u = 1/β is fitted if None, else held fixed.

    Temperature errors enter through the effective variance, as in fit_line. Returns
    (E0, c, u), their covariance (scaled up by chi^2/dof when that exceeds 1; zero row
    and column for a fixed u) and chi^2 per degree of freedom.
    """
    free_u = u is None
    p = np.array([eg.max() + 0.01, 1e-6, 0.0 if free_u else u])
    for _ in range(10):
        sigma = np.hypot(eg_err, varshni_slope(T, *p) * T_err)
        if free_u:
            p, cov = curve_fit(varshni, T, eg, p0=p, sigma=sigma, absolute_sigma=True)
        else:
            q, c2 = curve_fit(lambda t, e0, c: varshni(t, e0, c, u), T, eg, p0=p[:2],
                              sigma=sigma, absolute_sigma=True)
            p, cov = np.append(q, u), np.pad(c2, ((0, 1), (0, 1)))
    chi2_dof = np.sum(((eg - varshni(T, *p)) / sigma) ** 2) / (len(T) - len(p) + (not free_u))
    return p, cov * max(chi2_dof, 1.0), chi2_dof


def eg_gap(T):
    """Γ-valley Eg (eV) of GaP, InP and AlP at T (K) (Vurgaftman et al. 2001)."""
    gap = 2.886 + 0.1081 * (1 - 1 / np.tanh(164 / T))
    inp = 1.4236 - 0.363e-3 * T**2 / (T + 162)
    alp = 3.63 - 0.5771e-3 * T**2 / (T + 372)
    return gap, inp, alp


def eg_theory(T, x):
    """Eg (eV) of (Al_x Ga_1-x)_0.51 In_0.49 P, the method of Martikainen et al. (2020).

    Ternaries Ga/Al_0.51 In_0.49 P from the binaries with bowing +0.65 / -0.48 eV, then
    the quaternary from the ternaries with bowing 0.18 eV; bowing taken T-independent.
    """
    y = THEORY_GROUP_III_FRACTION
    gap, inp, alp = eg_gap(T)
    ga_in_p = y * gap + (1 - y) * inp - y * (1 - y) * 0.65
    al_in_p = y * alp + (1 - y) * inp + y * (1 - y) * 0.48
    return (1 - x) * ga_in_p + x * al_in_p - 0.18 * x * (1 - x)


def theory_al_fraction(peak_nm):
    """Al fraction x for which Eg(298 K) equals the datasheet peak photon energy."""
    e_peak = HC_EV_NM / (N_AIR * peak_nm)
    g, a = eg_theory(298.0, 0.0), eg_theory(298.0, 1.0)
    # Eg(298 K, x) = g + (a - g - 0.18) x + 0.18 x², solved for the root in [0, 1]
    return max(np.roots([0.18, a - g - 0.18, g - e_peak]).real, key=lambda r: 0 <= r <= 1)


def plot_eg_vs_t_varshni(T, T_err, eg, eg_err, path):
    """Eg against T with the Varshni fit with β free if β comes out positive; otherwise β → ∞
    and β fixed at the GaAs value."""
    p_free, cov_free, chi2_free = fit_varshni(T, T_err, eg, eg_err)
    u, u_err = p_free[2], np.sqrt(cov_free[2, 2])
    print("\nVarshni, Eg = E0 - α T²/(T + β), fitted as E0 - (α/β) T²/(1 + T/β):")
    print(f"  β free:  1/β = {u * 1e3:.2f} ± {u_err * 1e3:.2f} /1000 K  (β = {1 / u:.0f} K), "
          f"chi2/dof = {chi2_free:.1f}" + ("  -> β < 0, unphysical" if u + 2 * u_err < 0 else ""))

    fixed = [(np.inf, "β → ∞: $E_0 - (α/β)\\,T^2$"), (VARSHNI_BETA_GAAS_K, None)]
    for beta in VARSHNI_BETAS_K:
        p, cov, chi2 = fit_varshni(T, T_err, eg, eg_err, u=1 / beta)
        alpha = f"α = {p[1] * beta * 1e3:.3f} ± {np.sqrt(cov[1, 1]) * beta * 1e3:.3f} meV/K" \
            if np.isfinite(beta) else f"α/β = {p[1] * 1e6:.3f} ± {np.sqrt(cov[1, 1]) * 1e6:.3f} µeV/K²"
        print(f"  β = {beta:6.0f} K:  E0 = {p[0]:.4f} ± {np.sqrt(cov[0, 0]):.4f} eV,  {alpha},  "
              f"chi2/dof = {chi2:.1f}")

    fits = [(beta, label, *fit_varshni(T, T_err, eg, eg_err, u=1 / beta)) for beta, label in fixed]
    free_physical = u > 0
    if free_physical:
        fits = [(1 / u, f"β free, {1 / u:.0f} K", p_free, cov_free, chi2_free)]

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.errorbar(T, eg, xerr=T_err, yerr=eg_err, fmt="o", ms=5, color=BLUE, ecolor=BLUE,
                capsize=3, zorder=3, label="$E_g$ from spectrum fit")
    t_line = np.linspace(T.min() - 1, T.max() + 1, 200)

    # Theory, with the band from the ±2 nm datasheet peak tolerance
    x = theory_al_fraction(THEORY_PEAK_NM)
    x_lo, x_hi = (theory_al_fraction(THEORY_PEAK_NM + s * THEORY_PEAK_TOL_NM) for s in (1, -1))
    offset = np.mean(eg - eg_theory(T, x))
    theory_slope = (eg_theory(T.mean() + 0.5, x) - eg_theory(T.mean() - 0.5, x))
    print(f"\nTheory (Al_x Ga_1-x)_0.51 In_0.49 P, x = {x:.3f} ({x_lo:.3f}–{x_hi:.3f}):")
    print(f"  Eg({T.min():.1f} K) = {eg_theory(T.min(), x):.4f} eV, Eg({T.max():.1f} K) = "
          f"{eg_theory(T.max(), x):.4f} eV, dEg/dT({T.mean():.1f} K) = {theory_slope * 1e3:.3f} meV/K")
    print(f"  measured - theory = {offset * 1e3:.1f} meV on average")
    ax.fill_between(t_line, eg_theory(t_line, x_lo), eg_theory(t_line, x_hi), color=ORANGE,
                    alpha=0.15, lw=0, zorder=1)
    ax.plot(t_line, eg_theory(t_line, x), color=ORANGE, lw=2, zorder=2,
            label=f"Theory, (Al$_x$Ga$_{{1-x}}$)$_{{0.51}}$In$_{{0.49}}$P, x = {x:.3f}\n"
                  f"(band: peak {THEORY_PEAK_NM:g} ± {THEORY_PEAK_TOL_NM:g} nm, x = {x_lo:.3f}–{x_hi:.3f})")

    lines = [f"Theory: d$E_g$/dT({T.mean():.1f} K) = {theory_slope * 1e3:.3f} meV/K,",
             f"measured − theory = {offset * 1e3:.1f} meV (mean)"]
    # Fits listed under a "Fits with β ≥ 0" heading when β free is unphysical
    indent = "" if free_physical else "  "
    if not free_physical:
        lines += [f"β free: β = {1 / u:.0f} K (1/β = {u * 1e3:.1f} ± {u_err * 1e3:.1f} × 10⁻³ K⁻¹),"
                  " unphysical", "Fits with β ≥ 0:"]
    for i, ((beta, label, p, cov, chi2), ls, marker) in enumerate(zip(fits, ("-", "--"), ("o", "s"))):
        if i == 0 and free_physical:
            # α = c β; error from the covariance of c and u = 1/β
            jac = np.array([0, 1 / u, -p[1] / u**2])
            alpha_err = np.sqrt(jac @ cov @ jac)
            lines.append(f"β free: β = {beta:.0f} ± {u_err / u**2:.0f} K, $E_0$ = {p[0]:.4f} eV,")
            lines.append(f"  α = {p[1] * beta * 1e3:.3f} ± {alpha_err * 1e3:.3f} meV/K, "
                         f"χ²/dof = {chi2:.1f}")
        elif np.isfinite(beta):
            label = f"β fixed at GaAs value, {beta:.0f} K"
            lines.append(f"{indent}β = {beta:.0f} K: $E_0$ = {p[0]:.4f} eV, "
                         f"α = {p[1] * beta * 1e3:.3f} ± {np.sqrt(cov[1, 1]) * beta * 1e3:.3f} meV/K, "
                         f"χ²/dof = {chi2:.1f}")
        else:
            lines.append(f"{indent}β → ∞: $E_0$ = {p[0]:.4f} eV, "
                         f"α/β = {p[1] * 1e6:.3f} ± {np.sqrt(cov[1, 1]) * 1e6:.3f} µeV/K², "
                         f"χ²/dof = {chi2:.1f}")
        ax.plot(t_line, varshni(t_line, *p), color=INK, lw=1.5, ls=ls, zorder=4,
                label=f"Varshni, {label}")
        ax_res.errorbar(T, (eg - varshni(T, *p)) * 1e3, yerr=eg_err * 1e3, fmt=marker, ms=4,
                        mfc=BLUE if ls == "-" else "white", color=BLUE, ecolor=FAINT, capsize=2,
                        zorder=3, label="β → ∞" if not np.isfinite(beta) else
                        f"β = {beta:.0f} K" + (" (free)" if i == 0 and free_physical else ""))
    ax.set_title("LED bandgap against temperature: Varshni fit and theory", color=INK)
    ax.set_ylabel("$E_g$ (eV)", color=MUTED)
    style(ax)

    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.set_xlabel("Temperature (K)", color=MUTED)
    ax_res.set_ylabel("$E_g$ − fit (meV)", color=MUTED)
    ax_res.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    style(ax_res)

    fig.tight_layout()
    # Theory sits ~10-20 meV above the data, leaving no room inside the axes: legend and
    # results go to the right of the plot (bbox_inches="tight" widens the image for them)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
    ax.text(1.02, 0.45, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(spectra_dir, pattern, prefix, fit_ranges, plot_range, overlay_xlims, background=None):
    """Fit every spectrum matching pattern in spectra_dir; plots go to plots/spectra/<prefix>.

    fit_ranges(E_peak) gives the fit window followed by the alternative windows for the
    systematic error on Eg, plot_range(E_peak) the energy range plotted for that spectrum,
    and overlay_xlims the (linear, log) x limits of the overlay. background is an optional
    spectrometer background file subtracted from every spectrum.
    """
    experiment_dir = PLOTS_DIR / "spectra" / prefix
    files = sorted(spectra_dir.glob(pattern), key=lambda f: parse_temperature(f)[0])
    spectra = []
    print("   T (K)        Eg (eV)                  T_c (K)  γ (meV)  peak (eV)  rms/peak")
    for path in files:
        T, dT = parse_temperature(path)
        energy, intensity = load_spectrum(path, background)
        e_peak = energy[np.argmax(intensity)]
        fit_range, *alt_ranges = fit_ranges(e_peak)
        p, err, rms = fit_spectrum(energy, intensity, fit_range)
        alt = [fit_spectrum(energy, intensity, r)[0]["Eg"] for r in alt_ranges]
        egs = alt + [p["Eg"]]
        # Statistical error and half the spread over the alternative fit windows, in quadrature
        eg_err = np.hypot(err["Eg"], (max(egs) - min(egs)) / 2)
        fit = dict(p=p, Eg_err=eg_err, E_peak=e_peak)
        spectra.append(dict(T=T, dT=dT, E=energy, I=intensity, **fit))
        print(f"  {T:6.1f} ± {dT:.1f}  {p['Eg']:.4f} ± {eg_err:.4f} (stat {err['Eg']:.4f})"
              f"  {p['kTc'] / K_B_OVER_Q:6.1f}  {p['gamma'] * 1e3:6.2f}   {fit['E_peak']:.4f}"
              f"    {rms / intensity.max():.4f}")

        out_dir = experiment_dir / f"{prefix}_{T:.1f}K"
        out_dir.mkdir(parents=True, exist_ok=True)
        shown = plot_range(e_peak)
        plot_spectrum(energy, intensity, fit, T, dT, out_dir / f"{out_dir.name}_fit.png",
                      fit_range, shown)
        plot_spectrum_log(energy, intensity, fit, T, out_dir / f"{out_dir.name}_log.png",
                          fit_range, shown)

    overlay_dir = experiment_dir / f"{prefix}_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(spectra, overlay_dir / f"{overlay_dir.name}.png", *overlay_xlims)
    plot_overlay(spectra, overlay_dir / f"{overlay_dir.name}_normalised.png", *overlay_xlims,
                 normalise=True)

    eg_dir = experiment_dir / f"{prefix}_Eg_vs_T"
    eg_dir.mkdir(parents=True, exist_ok=True)
    T, T_err, eg_err = (np.array([s[k] for s in spectra]) for k in ("T", "dT", "Eg_err"))
    eg = np.array([s["p"]["Eg"] for s in spectra])
    plot_eg_vs_t(T, T_err, eg, eg_err, eg_dir / f"{eg_dir.name}.png")
    plot_eg_vs_t_varshni(T, T_err, eg, eg_err, eg_dir / f"{eg_dir.name}_Varshni.png")
    print(f"Plots -> {experiment_dir}")
    return spectra


def main():
    run(SPECTRA_DIR, "real_*.csv", PLOT_PREFIX,
        fit_ranges=lambda _: [FIT_RANGE_EV, *ALT_FIT_RANGES_EV],
        plot_range=lambda _: PLOT_RANGE_EV,
        overlay_xlims=((1.98, 2.10), (1.95, 2.13)))


if __name__ == "__main__":
    main()
