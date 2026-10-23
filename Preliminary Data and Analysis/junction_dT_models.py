"""Junction temperature rise ΔT against stage temperature from a heat balance, against linear.

The analyses take ΔT linear in T between the two pulse measurements
(dos_sigmoid_16Oct.JUNCTION_DT_K), as Martikainen et al. (2020) assume. Here ΔT instead
follows from the heat balance of the chip,

    ΔT(T) = R_th(T) P(T),    P = I Vf(T),    R_th ∝ T^n,

where P is the electrical power dissipated at 17 mA, from the measured Vf (Vf_vs_T.py; the
light emitted is neglected), and R_th the thermal resistance from junction to stage, with
the power law usual for conduction through solids. The two pulse points fix the scale of
R_th and n, so ΔT between them follows the measured Vf(T) instead of a straight line.

With w = ln(T/T1) / ln(T2/T1) this is ln ΔT = (1 - w) ln(ΔT1/V1) + w ln(ΔT2/V2) + ln V(T),
so the relative errors of ΔT1 and ΔT2 (taken as independent) combine with weights 1 - w, w.

Plots ΔT(T) of both models with P(T) and the implied R_th(T), and the Chen et al. (2009)
comparison of chen_model_16Oct.py again with this ΔT; the analyses themselves keep the
linear ΔT.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import BLUE, FAINT, GRID, INK, MUTED, ORANGE, PLOTS_DIR, style
from dos_sigmoid_16Oct import CURRENT_MA, JUNCTION_DT_K, SPECTRA_DIR
from spectrum_bandgap import parse_temperature
import chen_model_16Oct

VF_FILE = Path(__file__).with_name("data") / "LED_Vf_vs_T" / "16_Oct_Vf_17mA.csv"
# Vf above the last measured point is extrapolated with a line through the points above this
VF_EXTRAPOLATE_ABOVE_K = 240
OUTPUT_FILE = PLOTS_DIR / "pulses" / "junction_dT_models.png"
CHEN_OUTPUT_FILE = chen_model_16Oct.OUTPUT_DIR / "16_Oct_Chen_model_Tj_vs_T_heat_balance_dT.png"
# Colour of the heat-balance ΔT curve, apart from the blue and orange of the spectral points
HEAT_BALANCE_COLOR = "#1a9e5c"
STAGE_OUTPUT_FILE = chen_model_16Oct.OUTPUT_DIR / "16_Oct_Chen_model_Tj_vs_stage_T.png"


def forward_voltage(T):
    """Vf (V) at 17 mA at stage temperature T, interpolated in the measured table.

    Below the table Vf is held at its first value (it is flat below 79 K); above, it is
    extrapolated with a straight line through the points above VF_EXTRAPOLATE_ABOVE_K.
    """
    t, _, v, _ = np.loadtxt(VF_FILE, delimiter=",", skiprows=3, unpack=True)
    T = np.asarray(T, dtype=float)
    hot = t >= VF_EXTRAPOLATE_ABOVE_K
    return np.where(T > t[-1], np.polyval(np.polyfit(t[hot], v[hot], 1), T), np.interp(T, t, v))


def exponent():
    """n of R_th ∝ T^n, from the two measured ΔT."""
    (t1, dt1, _), (t2, dt2, _) = JUNCTION_DT_K
    return float(np.log((dt2 / forward_voltage(t2)) / (dt1 / forward_voltage(t1)))
                 / np.log(t2 / t1))


def dt_linear(T):
    """ΔT and its error, linear in T (as dos_sigmoid_16Oct.junction_temperature)."""
    (t1, dt1, e1), (t2, dt2, e2) = JUNCTION_DT_K
    w = (np.asarray(T, dtype=float) - t1) / (t2 - t1)
    return (1 - w) * dt1 + w * dt2, np.hypot((1 - w) * e1, w * e2)


def dt_heat_balance(T):
    """ΔT = R_th(T) I Vf(T) with R_th ∝ T^n through the two pulse points, and its error."""
    (t1, dt1, e1), (t2, dt2, e2) = JUNCTION_DT_K
    T = np.asarray(T, dtype=float)
    w = np.log(T / t1) / np.log(t2 / t1)
    dt = dt1 * forward_voltage(T) / forward_voltage(t1) * (T / t1) ** exponent()
    return dt, dt * np.hypot((1 - w) * e1 / dt1, w * e2 / dt2)


def plot_models(path):
    """ΔT(T) of both models, and P(T) with the R_th(T) of the heat balance."""
    T = np.linspace(JUNCTION_DT_K[0][0], JUNCTION_DT_K[1][0], 400)
    n = exponent()
    power_mw = CURRENT_MA * forward_voltage(T)

    fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for model, label, color, ls in (
            (dt_linear, "Linear in T (used in the analyses)", MUTED, "--"),
            (dt_heat_balance, rf"Heat balance: $R_{{th}} \propto T^{{{n:.2f}}}$ × $I V_f(T)$",
             BLUE, "-")):
        dt, err = model(T)
        ax.plot(T, dt, color=color, lw=1.8, ls=ls, zorder=3, label=label)
        ax.fill_between(T, dt - err, dt + err, color=color, alpha=0.15, lw=0, zorder=2)
    t_pts, dt_pts, e_pts = np.array(JUNCTION_DT_K).T
    ax.errorbar(t_pts, dt_pts, yerr=e_pts, fmt="o", ms=6, color=ORANGE, capsize=3, zorder=4,
                label="From pulses")
    ax.axhline(0, color=FAINT, lw=1)
    ax.set_ylabel("Junction rise above stage ΔT (K)", color=MUTED)
    ax.set_title(f"Junction temperature correction against stage temperature ({CURRENT_MA} mA)",
                 color=INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)

    ax_p.plot(T, power_mw, color=INK, lw=1.8, zorder=3, label=f"P = I $V_f$ ({CURRENT_MA} mA)")
    ax_p.set_ylabel("Dissipated power P (mW)", color=MUTED)
    ax_p.set_xlabel("Stage temperature (K)", color=MUTED)
    style(ax_p)
    ax_r = ax_p.twinx()
    dt, _ = dt_heat_balance(T)
    r_th = dt / (power_mw * 1e-3)
    ax_r.plot(T, r_th, color=BLUE, lw=1.8, ls="--", zorder=3, label=r"$R_{th}$ = ΔT / P")
    ax_r.set_yscale("log")
    ax_r.set_ylabel(r"$R_{th}$ (K/W, log scale)", color=MUTED)
    ax_r.spines["top"].set_visible(False)
    h, lab = ax_p.get_legend_handles_labels()
    h_r, lab_r = ax_r.get_legend_handles_labels()
    ax_p.legend(h + h_r, lab + lab_r, loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"n = {n:.2f}; R_th = {r_th[0]:.0f} K/W at {T[0]:.1f} K, "
          f"{r_th[-1]:.1f} K/W at {T[-1]:.1f} K")
    print(f"Saved {path}")


def plot_against_stage(results, path):
    """T_j - stage T against stage T: both ΔT models as curves, the Chen results as points.

    The lower panel repeats the two ΔT models alone, on a scale where they can be told apart.
    """
    T = np.linspace(JUNCTION_DT_K[0][0], JUNCTION_DT_K[1][0], 400)
    models = ((dt_linear, "ΔT linear in T", MUTED, "--"),
              (dt_heat_balance, rf"ΔT heat balance, $R_{{th}} \propto T^{{{exponent():.2f}}}$ × "
                                r"$I V_f(T)$", HEAT_BALANCE_COLOR, "-"))
    t_pts, dt_pts, e_pts = np.array(JUNCTION_DT_K).T
    stage = np.array([s["T"] for s in results])

    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for key, err_key, label, fmt, color, face in (
            ("Tm", "Tm_err", "Chen Eq. 10 (tail gradient) − stage T", "s", ORANGE, "white"),
            ("Tj", "Tj_err", "Chen Eq. 9 fit − stage T", "o", BLUE, BLUE)):
        ax.errorbar(stage, np.array([s[key] for s in results]) - stage,
                    yerr=[s[err_key] for s in results], fmt=fmt, ms=4, mfc=face, color=color,
                    ecolor=GRID, capsize=2, zorder=3, label=label)
    for a in (ax, ax_z):
        a.axhline(0, color=FAINT, lw=1, zorder=1)
        for model, label, color, ls in models:
            dt, err = model(T)
            a.fill_between(T, dt - err, dt + err, color=color, alpha=0.2, lw=0, zorder=4)
            a.plot(T, dt, color=color, lw=2.8, ls=ls, zorder=6, label=label)
        a.errorbar(t_pts, dt_pts, yerr=e_pts, fmt="D", ms=7, color=INK, mfc="white", capsize=3,
                   zorder=7, label="ΔT measured with pulses")
        a.set_ylabel("Junction − stage temperature (K)", color=MUTED)
        style(a)
    ax.set_title(f"Junction temperature correction against stage temperature, 16 Oct "
                 f"({CURRENT_MA} mA)", color=INK, fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax_z.set_title("(b) The two ΔT models alone", color=INK, fontsize=10)
    ax_z.set_ylim(-1, 12)
    ax_z.set_xlabel("Stage temperature (K)", color=MUTED)
    ax_z.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    print("  stage T (K)   Vf (V)   ΔT linear (K)   ΔT heat balance (K)")
    for T in (75.9, 80, 90, 100, 125, 150, 175, 200, 225, 250, 268.5, 294.5):
        (a, ae), (b, be) = dt_linear(T), dt_heat_balance(T)
        print(f"  {T:6.1f}       {float(forward_voltage(T)):.3f}    {float(a):5.2f} ± {float(ae):.2f}"
              f"     {float(b):5.2f} ± {float(be):.2f}")
    plot_models(OUTPUT_FILE)

    # The Chen et al. comparison again, with T_j from the pulses following the heat balance
    files = sorted(SPECTRA_DIR.glob("161026_*.csv"), key=lambda f: parse_temperature(f)[0])
    results = [chen_model_16Oct.analyse(f) for f in files]
    plot_against_stage(results, STAGE_OUTPUT_FILE)
    for s in results:
        dt, err = dt_heat_balance(s["T"])
        s["Tp"], s["Tp_err"] = s["T"] + float(dt), float(np.hypot(err, s["dT"]))
    tp = np.array([s["Tp"] for s in results])
    for key, name in (("Tm", "Eq. 10"), ("Tj", "Eq. 9")):
        d = np.array([s[key] for s in results]) - tp
        print(f"Chen {name}: mean (spectrum - pulses, heat balance ΔT) = {d.mean():+.1f} K, "
              f"rms = {np.sqrt(np.mean(d**2)):.1f} K")
    chen_model_16Oct.plot_summary(results, CHEN_OUTPUT_FILE,
                                  dt_label=f"ΔT = R_th I Vf, R_th ∝ T^{exponent():.2f}")
    print(f"Saved {CHEN_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
