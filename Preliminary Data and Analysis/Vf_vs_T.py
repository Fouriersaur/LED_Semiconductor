"""Forward voltage of the LED against temperature at a constant 17 mA, 16 Oct (76.7-268.5 K).

Plots Vf(T) with its error bars and, below it, the local slope dVf/dT between
neighbouring points, which converts a forward voltage drop into a junction temperature
rise (Martikainen et al., Rev. Sci. Instrum. 91, 015106 (2020), Sec. III).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import BLUE, INK, MUTED, PLOTS_DIR, style

DATA_FILE = Path(__file__).with_name("data") / "LED_Vf_vs_T" / "16_Oct_Vf_17mA.csv"
OUTPUT_FILE = PLOTS_DIR / "Vf_vs_T" / "16_Oct_Vf_vs_T_17mA.png"


def main():
    T, T_err, V, V_err = np.loadtxt(DATA_FILE, delimiter=",", skiprows=3, unpack=True)

    # Slope between neighbouring points, plotted at their mean temperature
    T_mid = (T[1:] + T[:-1]) / 2
    slope = np.diff(V) / np.diff(T)
    slope_err = np.abs(slope) * np.hypot(np.hypot(V_err[1:], V_err[:-1]) / np.diff(V),
                                         np.hypot(T_err[1:], T_err[:-1]) / np.diff(T))

    fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    ax.errorbar(T, V, xerr=T_err, yerr=V_err, fmt="o", color=BLUE, ms=4, capsize=2,
                zorder=3, label="Measured, 17 mA")
    ax.set_ylabel("Forward voltage $V_f$ / V")
    ax.set_title("LED forward voltage against temperature, 17 mA (16 Oct)", color=INK)
    ax.legend(frameon=False)
    style(ax)

    ax_s.errorbar(T_mid, slope * 1e3, yerr=slope_err * 1e3, fmt="o", color=BLUE, ms=4,
                  capsize=2, zorder=3)
    ax_s.axhline(0, color=MUTED, lw=1, zorder=2)
    ax_s.set_xlabel("Temperature / K")
    ax_s.set_ylabel("$dV_f/dT$ between neighbours / mV K$^{-1}$")
    style(ax_s)

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
