"""Junction temperature rise over a current pulse against stage temperature, 22 Oct (99.5-294.5 K).

Martikainen et al., Rev. Sci. Instrum. 91, 015106 (2020), Sec. III, measure the self-heating
ΔT of the junction over a 100 ms pulse at two heat sink temperatures and take it as linear
in T in between. Here the pulse is measured at seven stage temperatures to test that.

At each temperature the pulses (100 ms every 10 s, 17 mA) are averaged aligned at their
start, as in forward_voltage_t0.py, and Vf = V0 + b sqrt(t) is fitted to the averaged pulse.
The first ~1.5 ms carry a switch-on transient (a bump of a few tens of mV seen at every
temperature, also at 294.5 K where the heating is negligible), so the fit starts at
FIT_FROM_SQRT_MS; the spread of ΔV over ALT_FIT_FROM_SQRT_MS is taken as its systematic error.
The statistical error of b is the spread of b fitted to each pulse on its own.

The drop ΔV = V0 - Vf(end of pulse) = -b sqrt(t_end) is converted to a temperature rise with
the local slope dVf/dT of the 17 mA forward voltage table (Vf_vs_T.py), from a straight line
through the table points within SLOPE_HALF_WIDTH_K of the temperature; its spread over
ALT_SLOPE_HALF_WIDTHS_K and the slope between the two table points either side is the
slope's systematic error. Above the table (294.5 K) the
slope of the points above SLOPE_EXTRAPOLATE_ABOVE_K is used.

ΔT(T) is then fitted with a straight line, and the reduced χ² tells whether a line is
consistent with the points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import BLUE, FAINT, GRID, INK, MUTED, ORANGE, PLOTS_DIR, style
from forward_voltage_t0 import average_pulse, find_pulses, fit_sqrt_t

DATA_DIR = Path(__file__).with_name("data") / "LED_pulses" / "22 Oct Pulse Data"
VF_FILE = Path(__file__).with_name("data") / "LED_Vf_vs_T" / "16_Oct_Vf_17mA.csv"
OUTPUT_FILE = PLOTS_DIR / "pulses" / "22_Oct_pulse_dT_vs_T.png"
PULSES_OUTPUT_FILE = PLOTS_DIR / "pulses" / "22_Oct_averaged_pulses.png"

# Stage temperature (K) -> scope trace
FILES = {
    99.5: "99P5.CSV",
    134.7: "134P7.CSV",
    163.2: "163P2.CSV",
    185.4: "185P4.CSV",
    208.15: "208P15.CSV",
    240.1: "240P1.CSV",
    294.5: "294P5PU1 2.CSV",
}

FIT_FROM_SQRT_MS = 2.0
ALT_FIT_FROM_SQRT_MS = (1.5, 2.5, 3.0)
# The last samples of the averaged pulse sit on the falling edge
FIT_TO_SQRT_MS = 9.95
SLOPE_HALF_WIDTH_K = 8
ALT_SLOPE_HALF_WIDTHS_K = (5, 12, None)
SLOPE_EXTRAPOLATE_ABOVE_K = 240


def load_trace(path):
    """Time (s) and voltage (V) of a scope CSV; np.loadtxt is slow on 10^7 rows."""
    text = path.read_bytes().split(b"\n", 1)[1].replace(b"\r", b"").replace(b"\n", b",")
    t, v = np.fromstring(text.decode(), sep=",").reshape(-1, 2).T
    return t, v


def vf_slope(T, half_width=SLOPE_HALF_WIDTH_K):
    """dVf/dT (V/K) at 17 mA at T, and the standard error of the fitted slope.

    The line goes through the table points within half_width of T, or the three nearest
    where there are fewer (the table has a gap from 133.7 K to 157.2 K). half_width None
    gives the slope between the two points either side of T instead.
    """
    t, _, v, v_err = np.loadtxt(VF_FILE, delimiter=",", skiprows=3, unpack=True)
    if T > t[-1]:
        near = t >= SLOPE_EXTRAPOLATE_ABOVE_K
    elif half_width is None:
        k = np.searchsorted(t, T)
        return (v[k] - v[k - 1]) / (t[k] - t[k - 1]), np.hypot(v_err[k], v_err[k - 1]) / (t[k] - t[k - 1])
    else:
        near = np.abs(t - T) <= half_width
        if near.sum() < 3:
            near = np.isin(np.arange(len(t)), np.argsort(np.abs(t - T))[:3])
    p, cov = np.polyfit(t[near], v[near], 1, w=1 / v_err[near], cov="unscaled")
    return p[0], np.sqrt(cov[0, 0])


def fit_drop(t_avg, v_avg, x_from):
    """sqrt(t) fit of the averaged pulse from x_from to FIT_TO_SQRT_MS (sqrt(ms))."""
    x = np.sqrt(t_avg * 1e3)
    in_fit = (x >= x_from) & (x <= FIT_TO_SQRT_MS)
    return fit_sqrt_t(x[in_fit], v_avg[in_fit])


def analyse(T, name):
    t, v = load_trace(DATA_DIR / name)
    pulses = find_pulses(t, v)
    dt = np.median(np.diff(t[:1000]))
    t_avg, v_avg = average_pulse(t, v, pulses, dt)
    x_end = min(np.sqrt(t_avg[-1] * 1e3), FIT_TO_SQRT_MS)

    v0, b, *_ = fit_drop(t_avg, v_avg, FIT_FROM_SQRT_MS)
    drop = -b * x_end
    alt = [-fit_drop(t_avg, v_avg, x)[1] * x_end for x in ALT_FIT_FROM_SQRT_MS]
    drop_sys = (max(alt + [drop]) - min(alt + [drop])) / 2

    # Statistical error from the spread of b between pulses
    b_each = []
    for t0, idx in pulses:
        x = np.sqrt((t[idx] - t0) * 1e3)
        keep = (x >= FIT_FROM_SQRT_MS) & (x <= FIT_TO_SQRT_MS)
        b_each.append(fit_sqrt_t(x[keep], v[idx][keep])[1])
    drop_stat = np.std(b_each, ddof=1) / np.sqrt(len(b_each)) * x_end

    slope, slope_fit_err = vf_slope(T)
    alt_slopes = [vf_slope(T, w)[0] for w in ALT_SLOPE_HALF_WIDTHS_K]
    slope_sys = (max(alt_slopes + [slope]) - min(alt_slopes + [slope])) / 2
    slope_err = np.hypot(slope_fit_err, slope_sys)

    drop_err = np.hypot(drop_stat, drop_sys)
    dT = drop / -slope
    dT_err = abs(dT) * np.hypot(drop_err / drop, slope_err / slope) if drop else drop_err / -slope
    print(f"  {T:6.2f}   {len(pulses):2d}   {v0:.4f}   {drop * 1e3:6.2f} ± {drop_stat * 1e3:.2f} "
          f"± {drop_sys * 1e3:.2f}   {slope * 1e3:7.2f} ± {slope_err * 1e3:.2f}   "
          f"{dT:5.2f} ± {dT_err:.2f}")
    return dict(T=T, n=len(pulses), t_avg=t_avg, v_avg=v_avg, v0=v0, b=b, drop=drop,
                drop_err=drop_err, slope=slope, slope_err=slope_err, dT=dT, dT_err=dT_err)


def plot_pulses(results, path):
    """Averaged pulse of each temperature against sqrt(t), relative to its V0, with the fit."""
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
    for ax, r in zip(axs.ravel(), results):
        x = np.sqrt(r["t_avg"] * 1e3)
        dv = (r["v_avg"] - r["v0"]) * 1e3
        # Mean over bins of 0.1 sqrt(ms) so the trend shows through the 10 mV quantisation
        edges = np.arange(0, x[-1] + 0.1, 0.1)
        k = np.digitize(x, edges)
        x_bin = np.array([x[k == i].mean() for i in np.unique(k)])
        dv_bin = np.array([dv[k == i].mean() for i in np.unique(k)])
        ax.scatter(x, dv, s=1, color=FAINT, zorder=2, rasterized=True)
        ax.plot(x_bin, dv_bin, color=BLUE, lw=1.2, zorder=3, label="Binned mean")
        x_line = np.linspace(0, FIT_TO_SQRT_MS, 50)
        ax.plot(x_line, r["b"] * x_line * 1e3, color=ORANGE, lw=1.8, zorder=4,
                label=rf"Fit from {FIT_FROM_SQRT_MS:g} $\sqrt{{ms}}$")
        ax.axvspan(0, FIT_FROM_SQRT_MS, color=GRID, alpha=0.4, lw=0, zorder=1)
        ax.set_ylim(min(dv_bin.min(), r["b"] * 1e4) - 10, max(dv_bin.max(), 0) + 10)
        ax.set_title(f"{r['T']:g} K: ΔV = {r['drop'] * 1e3:.1f} mV, ΔT = {r['dT']:.2f} K",
                     color=INK, fontsize=10)
        style(ax)
    for ax in axs[-1]:
        ax.set_xlabel(r"Square root of time / $\sqrt{\mathrm{ms}}$")
    for ax in axs[:, 0]:
        ax.set_ylabel("$V_f - V_0$ / mV")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[-1, -1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dt(results, line, path):
    """ΔT against stage temperature with the straight-line fit and the two-point line."""
    T, dT, err = (np.array([r[k] for r in results]) for k in ("T", "dT", "dT_err"))
    (a, c), chi2_red = line

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.4]})
    T_line = np.linspace(T.min() - 5, T.max() + 5, 100)
    ax.plot(T_line, a + c * T_line, color=MUTED, lw=1.5, ls="--", zorder=2,
            label=f"Weighted straight line, χ²/dof = {chi2_red:.1f}")
    ends = [0, -1]
    two_point = np.interp(T_line, T[ends], dT[ends])
    ax.plot(T_line, two_point, color=ORANGE, lw=1.5, ls=":", zorder=2,
            label=f"Line through {T[0]:g} K and {T[-1]:g} K only (paper's assumption)")
    ax.errorbar(T, dT, yerr=err, fmt="o", ms=6, color=BLUE, capsize=3, zorder=3,
                label="From 100 ms pulses, 22 Oct")
    ax.axhline(0, color=FAINT, lw=1, zorder=1)
    ax.set_ylabel("Junction rise over pulse ΔT / K")
    ax.set_title("Self-heating over a 100 ms, 17 mA pulse against stage temperature",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    ax_res.errorbar(T, dT - (a + c * T), yerr=err, fmt="o", ms=5, color=BLUE, capsize=3,
                    zorder=3, label="− straight line")
    ax_res.errorbar(T + 2, dT - np.interp(T, T[ends], dT[ends]), yerr=err, fmt="s", ms=4,
                    mfc="white", color=ORANGE, capsize=3, zorder=3, label="− two-point line")
    ax_res.axhline(0, color=MUTED, lw=1, zorder=1)
    ax_res.set_xlabel("Stage temperature / K")
    ax_res.set_ylabel("Residual / K")
    ax_res.legend(frameon=False, fontsize=8)
    style(ax_res)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    print("  T (K)    n    V0 (V)   ΔV (mV) ± stat ± sys    dVf/dT (mV/K)    ΔT (K)")
    results = [analyse(T, name) for T, name in FILES.items()]
    T, dT, err = (np.array([r[k] for r in results]) for k in ("T", "dT", "dT_err"))

    (c, a), cov = np.polyfit(T, dT, 1, w=1 / err, cov="unscaled")
    chi2 = np.sum(((dT - (a + c * T)) / err) ** 2)
    dof = len(T) - 2
    print(f"\nStraight line: ΔT = {a:.2f} + {c * 1e3:.2f}e-3 T, χ²/dof = {chi2:.1f}/{dof} "
          f"= {chi2 / dof:.1f}")
    ends = np.interp(T, T[[0, -1]], dT[[0, -1]])
    print("Two-point line (99.5 K and 294.5 K) at the other temperatures, against measured:")
    for t_k, d, e, l in zip(T, dT, err, ends):
        print(f"  {t_k:6.2f} K: measured {d:5.2f} ± {e:.2f} K, line {l:5.2f} K, "
              f"difference {(d - l) / e:+.1f} σ")

    plot_pulses(results, PULSES_OUTPUT_FILE)
    plot_dt(results, ((a, c), chi2 / dof), OUTPUT_FILE)


if __name__ == "__main__":
    main()
