"""Joint density of states from the room-temperature LED spectrum, and the two-sigmoid DOS model.

Follows Vaskuri et al., J. Appl. Phys. 118, 203103 (2015), Sec. III and Fig. 3. With
Maxwell-Boltzmann occupation the emission is r(E) ∝ ρ(E) exp(-(E - (E_fn - E_fp)) / k T_j),
so dividing a spectrum taken at a known T_j by the Boltzmann factor, with E_fn - E_fp set to
the peak energy, gives the experimental DOS ρ(E) up to a constant. The model DOS is two
exponentially broadened steps (sigmoids), Eq. (4):

    ρ_est(E) = A1 / (1 + exp(-(E - Eg) / σ1)) + A2 / (1 + exp(-(E - Eg - ΔE2) / σ2))

fitted to ρ(E) by least squares on a log scale, Eq. (5).

Reference spectrum: the 12 spectra in data/LED_Vf_vs_T/room_temp_spectra (22 Oct, 293.5 K,
17 mA, 1.25 s each), averaged after subtracting the 16 Oct background (BACKGROUND) pixel by
pixel, as for the 16 Oct spectra. Their wavelengths are in air and the background's in
vacuum, but the pixels are the same. What offset is left is removed by the off-peak median
in load_spectrum. As in the paper the data are
averaged into 1 nm bins. Dividing by exp(-E/kT) amplifies the noise by e per kT above the
peak, so only bins above SIGNAL_TO_NOISE_MIN are used.

The Boltzmann factor takes the junction temperature, not the stage temperature. The
junction sits ΔT above the stage, measured from the forward voltage drop over a short pulse
(forward_voltage_t0.py, Martikainen et al., Rev. Sci. Instrum. 91, 015106 (2020), Sec. III)
with dVf/dT at 17 mA (Vf_vs_T.py) at two stage temperatures, and taken as linear in T in
between (JUNCTION_DT_K). The DOS of the 76.7 K spectrum is fitted too, for comparison with
the reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from IV_Curve_25C import BLUE, FAINT, GRID, INK, K_B_OVER_Q, MUTED, PLOTS_DIR, style
from spectrum_bandgap import BASELINE_OUTSIDE_EV, HC_EV_NM, TEMP_CMAP, load_spectrum, \
    parse_temperature

DATA_DIR = Path(__file__).with_name("data")
# 16 Oct temperature series, fitted with the calibrated model in dos_model_Tj_16Oct.py
SPECTRA_DIR = DATA_DIR / "LED_spectra_16Oct"
BACKGROUND = SPECTRA_DIR / "background_1610_1.csv"
# Reference spectra, averaged into the one spectrum the DOS is calibrated on
REFERENCE_DIR = DATA_DIR / "LED_Vf_vs_T" / "room_temp_spectra"
T_K = 293.5  # stage temperature of the reference
CURRENT_MA = 17
OUTPUT_DIR = PLOTS_DIR / "spectra" / "16_Oct_DOS"
PLOT_PREFIX = f"16_Oct_DOS_ref_{T_K:g}K"
# Lowest-temperature 16 Oct spectrum, whose DOS is also extracted and fitted
LOW_T_FILE = SPECTRA_DIR / "161026_76_7_pm0_2.csv"

# Junction temperature rise above the stage, (stage T, ΔT, error of ΔT) in K, from the pulse
# measurements; ΔT is linear in T through these two points
JUNCTION_DT_K = ((75.9, 10.1, 1.0), (294.5, 0.1, 0.2))

BIN_NM = 1.0
SIGNAL_TO_NOISE_MIN = 10
# Offsets (K) from T_j at which the DOS is also extracted, to show how it depends on the choice
SENSITIVITY_OFFSETS_K = (-20, 0, 20, 40)

PARAM_NAMES = ("lnA1", "Eg", "sigma1", "ln_A1_over_A2", "dE2", "sigma2")
# Starting ΔE2 (eV) and ln(A1/A2): the fit has local minima, so it is started from a grid
START_DE2_EV = (0.005, 0.01, 0.02, 0.04, 0.08, 0.12)
START_LN_RATIO = (-2.0, 0.0, 2.0)


def junction_temperature(T, T_err=0.0):
    """T_j and its error at stage temperature T (error T_err), with ΔT linear in T.

    The errors of ΔT at the two measured points are taken as independent and interpolated
    with the same weights as ΔT.
    """
    (t1, dt1, e1), (t2, dt2, e2) = JUNCTION_DT_K
    w = (T - t1) / (t2 - t1)
    return T + (1 - w) * dt1 + w * dt2, np.sqrt(((1 - w) * e1) ** 2 + (w * e2) ** 2 + T_err**2)


def averaged_spectrum(paths, background=None):
    """Energy and mean intensity per eV of spectra on one wavelength grid."""
    spectra = [load_spectrum(path, background) for path in paths]
    energy = spectra[0][0]
    if not all(np.allclose(e, energy) for e, _ in spectra):
        raise ValueError("the reference spectra are not on one wavelength grid")
    return energy, np.mean([i for _, i in spectra], axis=0)


def binned_spectrum(path, background, bin_nm=BIN_NM):
    """binned_spectrum_from of one spectrum file."""
    return binned_spectrum_from(*load_spectrum(path, background), bin_nm)


def binned_spectrum_from(energy, intensity, bin_nm=BIN_NM):
    """Energy, intensity per eV (normalised to the peak) and its noise, in bins of bin_nm.

    The noise per bin is the rms of the off-peak signal divided by √(pixels in the bin).
    """
    intensity = intensity / intensity.max()
    off_peak = (energy < BASELINE_OUTSIDE_EV[0]) | (energy > BASELINE_OUTSIDE_EV[1])
    noise = np.std(intensity[off_peak])
    index = np.floor(HC_EV_NM / energy / bin_nm).astype(int)
    bins = np.unique(index)
    e = np.array([energy[index == k].mean() for k in bins])
    i = np.array([intensity[index == k].mean() for k in bins])
    n = np.array([np.sum(index == k) for k in bins])
    order = np.argsort(e)
    return e[order], i[order], (noise / np.sqrt(n))[order]


def signal_window(intensity, noise, snr_min=SIGNAL_TO_NOISE_MIN):
    """Mask of the contiguous run of bins around the peak with intensity / noise ≥ snr_min."""
    peak = np.argmax(intensity)
    below = np.flatnonzero(intensity / noise < snr_min)
    lo = below[below < peak].max() + 1 if np.any(below < peak) else 0
    hi = below[below > peak].min() if np.any(below > peak) else len(intensity)
    used = np.zeros_like(intensity, dtype=bool)
    used[lo:hi] = True
    return used


def experimental_dos(energy, intensity, e_peak, T):
    """ρ(E) = r(E) / exp(-(E - E_peak) / kT), with E_fn - E_fp = E_peak."""
    return intensity * np.exp((energy - e_peak) / (K_B_OVER_Q * T))


def sigmoid(x):
    return 0.5 * (1 + np.tanh(x / 2))


def model_dos(energy, ln_a1, eg, sigma1, ln_ratio, de2, sigma2):
    """Eq. (4), with A1 = exp(ln_a1) and A2 = A1 / exp(ln_ratio)."""
    a1 = np.exp(ln_a1)
    return a1 * sigmoid((energy - eg) / sigma1) \
        + a1 * np.exp(-ln_ratio) * sigmoid((energy - eg - de2) / sigma2)


def fit_dos(energy, dos):
    """Least squares on log ρ, Eq. (5). Returns the parameters, their errors and the rms."""
    log_dos = np.log(dos)

    def resid(p):
        return np.log(model_dos(energy, *p)) - log_dos

    e_mid = energy[np.argmin(np.abs(log_dos - (log_dos.max() + log_dos.min()) / 2))]
    starts = [[log_dos.max(), e_mid, 0.004, r, de2, 0.006]
              for de2 in START_DE2_EV for r in START_LN_RATIO]
    lower = [-50, energy.min(), 2e-4, -10, 0.0, 2e-4]
    upper = [50, energy.max(), 0.1, 10, 0.3, 0.1]
    scale = [1, 1e-3, 1e-3, 1, 1e-3, 1e-3]
    best = min((least_squares(resid, p0, bounds=(lower, upper), x_scale=scale) for p0 in starts),
               key=lambda r: r.cost)
    dof = len(energy) - len(best.x)
    rms = np.sqrt(2 * best.cost / dof)
    # Unweighted fit: errors from the Jacobian with the residual variance as the noise
    cov = np.linalg.pinv(best.jac.T @ best.jac) * rms**2
    return best.x, np.sqrt(np.diag(cov)), rms


def describe(p, err):
    return [f"$E_g$ = {p[1]:.4f} ± {err[1]:.4f} eV",
            f"σ1 = {p[2] * 1e3:.2f} ± {err[2] * 1e3:.2f} meV",
            f"$A_1/A_2$ = {np.exp(p[3]):.3g} (ln: {p[3]:.2f} ± {err[3]:.2f})",
            f"Δ$E_2$ = {p[4] * 1e3:.1f} ± {err[4] * 1e3:.1f} meV",
            f"σ2 = {p[5] * 1e3:.2f} ± {err[5] * 1e3:.2f} meV"]


def ideal_dos(energy, p):
    """Unbroadened steps at Eg and Eg + ΔE2 with the fitted heights (dotted line in Fig. 3)."""
    a1 = np.exp(p[0])
    return a1 * (energy > p[1]) + a1 * np.exp(-p[3]) * (energy > p[1] + p[4])


def plot_dos_fit(c, name, path):
    """Like Fig. 3 of the paper: DOS with the model on linear and log scales, and residuals.

    c is a dos_and_fit result and name what the spectrum is, for the title.
    """
    e, dos, dos_err, e_peak = c["e"][c["used"]], c["dos"], c["dos_err"], c["e_peak"]
    p2, err2, rms2 = c["fit"]
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_lin, ax_log = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_res = fig.add_subplot(grid[1, :])
    e_line = np.linspace(e.min(), e.max(), 1000)
    for ax in (ax_lin, ax_log):
        ax.errorbar(e, dos, yerr=dos_err, fmt="o", ms=4, mfc="white", color=BLUE, ecolor=FAINT,
                    capsize=0, zorder=3, label="Experimental DOS")
        ax.plot(e_line, model_dos(e_line, *p2), color=INK, lw=1.5, zorder=4,
                label="Two sigmoids, Eq. (4)")
        ax.plot(e_line, ideal_dos(e_line, p2), color=MUTED, lw=1, ls=":", zorder=2,
                label="Ideal steps")
        ax.set_xlabel("Photon energy E (eV)", color=MUTED)
        style(ax)
    ax_lin.set_ylim(0, 1.15 * max(dos.max(), model_dos(e_line, *p2).max()))
    ax_lin.set_ylabel("DOS (arb. units)", color=MUTED)
    ax_lin.set_title("(a) Linear scale", color=INK)
    ax_lin.legend(loc="upper left", fontsize=8, frameon=False)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("DOS (arb. units, log scale)", color=MUTED)
    ax_log.set_title("(b) Log scale", color=INK)
    lines = [f"Stage {c['T_stage']:g} K, $T_j$ = {c['T']:.1f} ± {c['T_err']:.1f} K, "
             f"I = {CURRENT_MA} mA",
             f"$E_{{fn}} - E_{{fp}}$ = $E_{{peak}}$ = {e_peak:.4f} eV",
             *describe(p2, err2), f"rms(ln ρ) = {rms2:.3f}"]
    ax_log.text(0.97, 0.03, "\n".join(lines), transform=ax_log.transAxes, va="bottom",
                ha="right", fontsize=8, color=INK, bbox=dict(facecolor="white", edgecolor=GRID))

    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.plot(e, np.log(dos) - np.log(model_dos(e, *p2)), "o", ms=4, color=INK)
    ax_res.set_xlabel("Photon energy E (eV)", color=MUTED)
    ax_res.set_ylabel("ln ρ − ln ρ$_{est}$", color=MUTED)
    style(ax_res)

    fig.suptitle(f"Joint DOS from the {name} at {c['T_stage']:g} K stage "
                 f"($T_j$ = {c['T']:.1f} K), {CURRENT_MA} mA "
                 "(method of Vaskuri et al. 2015, Fig. 3)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_temperature_sensitivity(c, path):
    """Experimental DOS extracted with different assumed T around T_j, normalised at the peak."""
    e, intensity, e_peak, used = c["e"], c["intensity"], c["e_peak"], c["used"]
    fig, ax = plt.subplots(figsize=(7, 5))
    temps = [c["T"] + dt for dt in SENSITIVITY_OFFSETS_K]
    for dt, T in zip(SENSITIVITY_OFFSETS_K, temps):
        dos = experimental_dos(e[used], intensity[used], e_peak, T)
        color = TEMP_CMAP((T - min(temps)) / (max(temps) - min(temps)) * 0.8 + 0.2)
        ax.plot(e[used], dos, "o-", ms=3, lw=1, color=color,
                label=f"T = {T:.1f} K" + (" ($T_j$)" if dt == 0 else ""))
    ax.axvline(e_peak, color=MUTED, lw=1, ls="--", label=f"$E_{{peak}}$ = {e_peak:.4f} eV")
    ax.set_yscale("log")
    ax.set_xlabel("Photon energy E (eV)", color=MUTED)
    ax.set_ylabel("DOS = r(E) exp((E − $E_{peak}$)/kT) (arb. units)", color=MUTED)
    ax.set_title(f"Experimental DOS against the assumed temperature "
                 f"(stage {c['T_stage']:g} K)", color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def dos_and_fit(e, intensity, noise, T_stage, T_stage_err=0.0):
    """Fit window, E_peak, the DOS at the junction temperature and its two-sigmoid fit."""
    T, T_err = junction_temperature(T_stage, T_stage_err)
    e_peak = e[np.argmax(intensity)]
    used = signal_window(intensity, noise)
    dos = experimental_dos(e[used], intensity[used], e_peak, T)
    dos_err = dos * noise[used] / intensity[used]
    # Normalise so the fitted first step is of order 1 (only ratios matter)
    dos, dos_err = dos / dos.max(), dos_err / dos.max()
    return dict(e=e, intensity=intensity, noise=noise, used=used, e_peak=e_peak, dos=dos,
                dos_err=dos_err, fit=fit_dos(e[used], dos), T_stage=T_stage, T=T, T_err=T_err)


def calibration():
    """dos_and_fit of the averaged reference spectrum."""
    paths = sorted(REFERENCE_DIR.glob("*.csv"))
    e, intensity, noise = binned_spectrum_from(*averaged_spectrum(paths, BACKGROUND))
    return dict(dos_and_fit(e, intensity, noise, T_K), n_spectra=len(paths))


def low_temperature():
    """dos_and_fit of the 76.7 K spectrum."""
    T_stage, T_stage_err = parse_temperature(LOW_T_FILE)
    return dos_and_fit(*binned_spectrum(LOW_T_FILE, BACKGROUND), T_stage, T_stage_err)


def plot_dos_against_temperature(c, path):
    """DOS of every 16 Oct spectrum, extracted at its junction T, against E - E_peak.

    Each is normalised at the peak. Shows how far above the peak each spectrum reaches before
    the noise, and where the DOS rises or falls; the reference is drawn on top.
    """
    files = sorted(SPECTRA_DIR.glob("161026_*.csv"), key=lambda f: parse_temperature(f)[0])
    temps = [junction_temperature(parse_temperature(f)[0])[0] for f in files]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for f, T in zip(files, temps):
        e, intensity, noise = binned_spectrum(f, BACKGROUND)
        used = signal_window(intensity, noise)
        e_peak = e[np.argmax(intensity)]
        dos = experimental_dos(e[used], intensity[used], e_peak, T)
        color = TEMP_CMAP(0.2 + 0.8 * (T - min(temps)) / (max(temps) - min(temps)))
        ax.plot((e[used] - e_peak) * 1e3, dos, lw=1, color=color)
    used = c["used"]
    dos = experimental_dos(c["e"][used], c["intensity"][used], c["e_peak"], c["T"])
    ax.plot((c["e"][used] - c["e_peak"]) * 1e3, dos, color=INK, lw=2.5,
            label=f"Reference, $T_j$ = {c['T']:.1f} K (22 Oct)")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 30)
    ax.set_xlabel("E − $E_{peak}$ (meV)", color=MUTED)
    ax.set_ylabel("DOS = r(E) exp((E − $E_{peak}$)/k$T_j$)", color=MUTED)
    ax.set_title("Experimental DOS of each spectrum at its own junction temperature", color=INK)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style(ax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(min(temps), max(temps)),
        cmap=TEMP_CMAP.from_list("t", [TEMP_CMAP(0.2), TEMP_CMAP(1.0)])), ax=ax, pad=0.02)
    cbar.set_label("Junction temperature, 16 Oct (K)", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report(c):
    """Print the two-sigmoid fit of c, and the same fit at other assumed temperatures."""
    e, intensity, used, e_peak = c["e"], c["intensity"], c["used"], c["e_peak"]
    print(f"Stage {c['T_stage']:g} K -> T_j = {c['T']:.2f} ± {c['T_err']:.2f} K, "
          f"{CURRENT_MA} mA, E_peak = {e_peak:.4f} eV")
    print(f"Fit range {e[used].min():.4f}–{e[used].max():.4f} eV "
          f"({used.sum()} bins of {BIN_NM:g} nm, S/N ≥ {SIGNAL_TO_NOISE_MIN})")
    p, err, rms = c["fit"]
    print(f"Two sigmoids: rms(ln ρ) = {rms:.3f}")
    for line in describe(p, err):
        print("  " + line.replace("$", "").replace("_", ""))

    # Same two-sigmoid fit at other assumed temperatures, to show what T does to the DOS
    print("\nTwo-sigmoid fit against the assumed T:")
    print("   T (K)   Eg (eV)   σ1 (meV)  A1/A2    ΔE2 (meV)  σ2 (meV)  rms(ln ρ)")
    for dt in SENSITIVITY_OFFSETS_K:
        T = c["T"] + dt
        d = experimental_dos(e[used], intensity[used], e_peak, T)
        p, _, rms = fit_dos(e[used], d / d.max())
        print(f"  {T:6.1f}  {p[1]:.4f}   {p[2] * 1e3:6.2f}   {np.exp(p[3]):8.3g}  "
              f"{p[4] * 1e3:7.1f}    {p[5] * 1e3:6.2f}    {rms:.3f}")


def main():
    c = calibration()
    print(f"Reference: mean of {c['n_spectra']} spectra in {REFERENCE_DIR.name}")
    report(c)
    low = low_temperature()
    print(f"\n{LOW_T_FILE.name}:")
    report(low)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dos_fit(c, "reference LED spectrum", OUTPUT_DIR / f"{PLOT_PREFIX}_sigmoid_fit.png")
    plot_temperature_sensitivity(c, OUTPUT_DIR / f"{PLOT_PREFIX}_vs_assumed_T.png")
    low_prefix = f"16_Oct_DOS_{low['T_stage']:g}K"
    plot_dos_fit(low, "LED spectrum", OUTPUT_DIR / f"{low_prefix}_sigmoid_fit.png")
    plot_temperature_sensitivity(low, OUTPUT_DIR / f"{low_prefix}_vs_assumed_T.png")
    plot_dos_against_temperature(c, OUTPUT_DIR / "16_Oct_DOS_all_T_vs_E_minus_Epeak.png")
    print(f"\nPlots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
