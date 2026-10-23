"""Junction temperature from the slope of the high-energy tail of each 16 Oct spectrum.

Above the peak the band-to-band emission falls as the Boltzmann factor of the carriers,
I(E) ∝ ρ(E) exp(-E / k T_j), so the slope of ln I(E) on the high-energy side gives T_j
(Vaitonis, Vitta and Žukauskas, J. Appl. Phys. 103, 093110 (2008); reference 8 of
Martikainen et al. 2020). Two versions are fitted to the 1 nm binned spectra, weighted by
the noise of ln I:

    plain:     ln I = c - E / k T_j
    with DOS:  ln I = c + ½ ln(E - Eg) - E / k T_j

The second takes the √(E - Eg) joint DOS of spectrum_bandgap.py into account, with Eg from
that fit; the DOS rises over the tail and makes the plain slope too shallow, so the plain
T_j comes out higher. Neither includes the broadening (γ of the sech in spectrum_bandgap.py),
which also flattens the tail and matters most when γ is close to kT, at low temperature.

The tail is the run of bins above the peak whose intensity is between TAIL_FRACTION[1]
and TAIL_FRACTION[0] of the peak and at least SIGNAL_TO_NOISE_MIN times the noise. Half
the spread of T_j over ALT_TAIL_FRACTIONS is taken as the systematic error from the choice
of window. T_j is compared with the stage temperature plus the ΔT from the pulse
measurements (dos_sigmoid_16Oct.junction_temperature).
"""

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import BLUE, FAINT, INK, K_B_OVER_Q, MUTED, ORANGE, PLOTS_DIR, style
from spectrum_bandgap import fit_spectrum, load_spectrum, parse_temperature
from spectrum_bandgap_16Oct import FIT_WINDOW_EV
from dos_sigmoid_16Oct import (BACKGROUND, CURRENT_MA, SIGNAL_TO_NOISE_MIN, SPECTRA_DIR,
                               binned_spectrum, junction_temperature)

OUTPUT_DIR = PLOTS_DIR / "spectra" / "16_Oct_high_energy_tail"
# Tail window as fractions of the peak (upper, lower), and alternatives for the systematic error
TAIL_FRACTION = (0.3, 0.01)
ALT_TAIL_FRACTIONS = ((0.5, 0.01), (0.3, 0.03), (0.1, 0.005), (0.5, 0.05))
# Spectrum shown in its own plot as well as in the summary
SHOWN_T_K = 76.7


def tail_window(intensity, noise, fraction):
    """Mask of the contiguous run of bins above the peak inside fraction = (upper, lower)."""
    peak = np.argmax(intensity)
    used = np.zeros_like(intensity, dtype=bool)
    for k in range(peak + 1, len(intensity)):
        if intensity[k] < fraction[1] or intensity[k] < SIGNAL_TO_NOISE_MIN * noise[k]:
            break
        used[k] = intensity[k] <= fraction[0]
    return used


def fit_tail(e, intensity, noise, used, eg=None):
    """Weighted straight line of ln I (minus ½ ln(E - Eg) if eg is given) against E.

    Returns T_j and its error, and the line's intercept and slope (per eV).
    """
    x, y = e[used], np.log(intensity[used])
    if eg is not None:
        y = y - 0.5 * np.log(x - eg)
    w = intensity[used] / noise[used]  # 1 / σ of ln I
    design = np.column_stack([np.ones_like(x), x]) * w[:, None]
    p, res, *_ = np.linalg.lstsq(design, y * w, rcond=None)
    # Errors scaled by the reduced χ², which also covers the model not being exact
    chi2 = res[0] / (len(x) - 2) if len(res) and len(x) > 2 else 1.0
    cov = np.linalg.inv(design.T @ design) * max(chi2, 1.0)
    tj = -1 / (K_B_OVER_Q * p[1])
    return tj, tj * np.sqrt(cov[1, 1]) / abs(p[1]), p


def analyse(path):
    T, dT = parse_temperature(path)
    tc, tc_err = junction_temperature(T, dT)
    e, intensity, noise = binned_spectrum(path, BACKGROUND)
    energy, raw = load_spectrum(path, BACKGROUND)
    e_peak = energy[np.argmax(raw)]
    eg = fit_spectrum(energy, raw, (e_peak + FIT_WINDOW_EV[0], e_peak + FIT_WINDOW_EV[1]))[0]["Eg"]

    result = dict(T=T, dT=dT, Tc=tc, Tc_err=tc_err, e=e, I=intensity, noise=noise, Eg=eg)
    for name, g in (("plain", None), ("dos", eg)):
        used = tail_window(intensity, noise, TAIL_FRACTION)
        tj, err, line = fit_tail(e, intensity, noise, used, g)
        alt = [fit_tail(e, intensity, noise, tail_window(intensity, noise, f), g)[0]
               for f in ALT_TAIL_FRACTIONS]
        sys = (max(alt + [tj]) - min(alt + [tj])) / 2
        result[name] = dict(Tj=tj, stat=err, sys=sys, err=np.hypot(err, sys), line=line,
                            used=used)
    return result


def plot_tail(r, path):
    """The spectrum on a log scale with the two tail fits, and ln I minus the fit below."""
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    e, intensity, noise = r["e"], r["I"], r["noise"]
    used = r["plain"]["used"]
    shown = intensity > 0
    ax.errorbar(e[shown & ~used], intensity[shown & ~used], yerr=noise[shown & ~used],
                fmt="o", ms=3, mfc="white", color=FAINT, ecolor=FAINT, zorder=2,
                label="Measured (1 nm bins)")
    ax.errorbar(e[used], intensity[used], yerr=noise[used], fmt="o", ms=3.5, color=BLUE,
                ecolor=BLUE, zorder=3, label="In tail fit")
    e_line = np.linspace(e[np.argmax(intensity)], e[used].max() + 0.02, 200)
    c, b = r["plain"]["line"]
    ax.plot(e_line, np.exp(c + b * e_line), color=INK, lw=1.5, zorder=4,
            label=f"$e^{{-E/kT_j}}$: $T_j$ = {r['plain']['Tj']:.1f} ± {r['plain']['err']:.1f} K")
    c, b = r["dos"]["line"]
    ax.plot(e_line, np.exp(c + b * e_line) * np.sqrt(e_line - r["Eg"]), color=ORANGE, lw=1.5,
            ls="--", zorder=4,
            label=rf"$\sqrt{{E-E_g}}\,e^{{-E/kT_j}}$: $T_j$ = {r['dos']['Tj']:.1f} ± "
                  f"{r['dos']['err']:.1f} K")
    ax.set_yscale("log")
    ax.set_ylim(noise.min() * 2, 2)
    ax.set_xlim(e[np.argmax(intensity)] - 0.06, e[used].max() + 0.04)
    ax.set_ylabel("Intensity / peak (log scale)", color=MUTED)
    ax.set_title(f"High-energy tail at {r['T']:.1f} K stage ($T_j$ = {r['Tc']:.1f} ± "
                 f"{r['Tc_err']:.1f} K from pulses), {CURRENT_MA} mA", color=INK, fontsize=10)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    style(ax)

    c, b = r["plain"]["line"]
    ax_res.axhline(0, color=MUTED, lw=1)
    ax_res.errorbar(e[used], np.log(intensity[used]) - (c + b * e[used]),
                    yerr=noise[used] / intensity[used], fmt="o", ms=3, color=BLUE, ecolor=FAINT)
    ax_res.set_ylabel("ln I − fit", color=MUTED)
    ax_res.set_xlabel("Photon energy E (eV)", color=MUTED)
    style(ax_res)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(results, path):
    """T_j from the tail against T_j from the pulses, and their difference."""
    tc = np.array([r["Tc"] for r in results])
    tc_err = np.array([r["Tc_err"] for r in results])
    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    lim = (tc.min() - 5, tc.max() + 5)
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=2, label="Tail = from pulses")
    for name, label, fmt, color in (("plain", r"$e^{-E/kT_j}$", "o", BLUE),
                                    ("dos", r"$\sqrt{E-E_g}\,e^{-E/kT_j}$", "s", ORANGE)):
        tj = np.array([r[name]["Tj"] for r in results])
        err = np.array([r[name]["err"] for r in results])
        for a, y in ((ax, tj), (ax_d, tj - tc)):
            a.errorbar(tc, y, xerr=tc_err, yerr=err, fmt=fmt, ms=4, mfc="white" if fmt == "s"
                       else color, color=color, ecolor=FAINT, capsize=2, zorder=3,
                       label=f"Tail fit, {label}")
    ax.set_ylabel("$T_j$ from the high-energy tail (K)", color=MUTED)
    ax.set_title(f"Junction temperature from the high-energy tail, 16 Oct ({CURRENT_MA} mA)",
                 color=INK)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax)
    ax_d.axhline(0, color=MUTED, lw=1)
    ax_d.set_xlabel("$T_j$ from pulses: stage T + ΔT (K)", color=MUTED)
    ax_d.set_ylabel("Tail − from pulses (K)", color=MUTED)
    style(ax_d)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    files = sorted(SPECTRA_DIR.glob("161026_*.csv"), key=lambda f: parse_temperature(f)[0])
    results = [analyse(f) for f in files]

    print(f"Tail window {TAIL_FRACTION[0]:g}–{TAIL_FRACTION[1]:g} of the peak "
          f"(alternatives {ALT_TAIL_FRACTIONS})")
    print("  stage T (K)   Tj pulses (K)   bins   Tj plain (K)              "
          "Tj with DOS (K)           Eg (eV)")
    for r in results:
        p, d = r["plain"], r["dos"]
        print(f"  {r['T']:6.1f} ± {r['dT']:.1f}  {r['Tc']:6.1f} ± {r['Tc_err']:.1f}    "
              f"{p['used'].sum():3d}   {p['Tj']:6.1f} ± {p['stat']:.1f} ± {p['sys']:4.1f}     "
              f"{d['Tj']:6.1f} ± {d['stat']:.1f} ± {d['sys']:4.1f}     {r['Eg']:.4f}")
    tc = np.array([r["Tc"] for r in results])
    for name in ("plain", "dos"):
        diff = np.array([r[name]["Tj"] for r in results]) - tc
        print(f"{name}: mean (tail - pulses) = {diff.mean():+.1f} K, "
              f"rms = {np.sqrt(np.mean(diff**2)):.1f} K")

    fits_dir = OUTPUT_DIR / "16_Oct_high_energy_tail_fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        plot_tail(r, fits_dir / f"16_Oct_high_energy_tail_{r['T']:.1f}K.png")
    shown = min(results, key=lambda r: abs(r["T"] - SHOWN_T_K))
    plot_tail(shown, OUTPUT_DIR / f"16_Oct_high_energy_tail_{shown['T']:.1f}K.png")
    plot_summary(results, OUTPUT_DIR / "16_Oct_high_energy_tail_Tj_vs_T.png")
    print(f"\nPlots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
