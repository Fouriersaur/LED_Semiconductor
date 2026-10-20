"""Forward voltage at the start of each current pulse, from scope traces of the LED voltage.

The LED is driven by ~100 ms current pulses every 10 s. During each pulse the junction
heats and the forward voltage drops, so Vf at t = 0 (start of the pulse) is the value at
the heat sink temperature. As in Martikainen et al., Rev. Sci. Instrum. 91, 015106 (2020)
(Fig. 4) and CIE 225:2017, Vf is fitted as a straight line against sqrt(t) and back
extrapolated to t = 0, skipping the first samples of each pulse, which are disturbed by
the switch-on.

The pulse start t0 is where the rising edge crosses halfway from the baseline to the flat
top, interpolated between samples. The fit runs over the flat top of the pulse (Vf above
PLATEAU_FRACTION of its level), minus its first `skip` samples and the last one, which
can sit on the falling edge. The spread of Vf(0) over `alt_skips` is taken as the
systematic error from the choice of skipped samples. Both default to SKIP_SAMPLES and
ALT_SKIP_SAMPLES and are set per dataset for faster sampled traces, where the switch-on
overshoot spans more samples.

Each pulse is fitted on its own, since Vf can drift between pulses as the cryostat warms.
As in Fig. 4, the pulses are also averaged sample by sample (aligned at t0) and the
averaged pulse is fitted and back extrapolated the same way.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from IV_Curve_25C import BLUE, FAINT, INK, MUTED, ORANGE, PLOTS_DIR, style

DATA_DIR = Path(__file__).with_name("data") / "LED_pulses"
OUTPUT_DIR = PLOTS_DIR / "pulses"

# Prefix of the output plots -> scope traces (pooled when there are several), and
# optionally the samples skipped at the start of each pulse
DATASETS = {
    "forward_voltage_t0": dict(files=["TRIGGER.CSV"]),
    "forward_voltage_t0_83K": dict(files=["83KPUL03.CSV"]),
    "forward_voltage_t0_75p9K_run1": dict(files=["75P9PUL1.CSV"]),
    "forward_voltage_t0_75p9K_run2": dict(files=["75P9PUL2.CSV"]),
    # 0.46 ms sampling, one measurement split over several files (there is no 75P9PU6)
    "forward_voltage_t0_75p9K_fast": dict(files=[f"75P9PU{k}.CSV" for k in (1, 2, 3, 4, 5, 7)]),
    # 24 µs sampling: the rising edge and its overshoot take ~0.15 ms, so skip 0.24 ms
    # (alternatives 0.12-0.96 ms)
    # The averaged pulse also drops steeply over the first ~2 ms, so that early part is
    # fitted on its own too (window in sqrt(ms)), as in Fig. 4(b) of the paper
    "forward_voltage_t0_75p9K_PU08": dict(
        files=["75P9PU08.CSV"], skip=10, alt_skips=(5, 20, 40),
        early_window=(0.55, 1.4),
        alt_early_windows=((0.5, 1.4), (0.6, 1.4), (0.55, 1.2), (0.55, 1.6))),
}

PLATEAU_FRACTION = 0.95
SKIP_SAMPLES = 2
ALT_SKIP_SAMPLES = (1, 3, 4)


def load_traces(paths):
    """Time and voltage of the traces in paths one after the other, and each one's start.

    Each trace is shifted to start 1 s after the previous one ends, with a NaN sample in
    between, so a pulse cut off at the end of one file is not joined to the next.
    """
    ts, vs, starts, end = [], [], [], 0.0
    for path in paths:
        t, v = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        t = t - t[0] + end
        ts += [t, [t[-1] + 0.5]]
        vs += [v, [np.nan]]
        starts.append(t[0])
        end = t[-1] + 1.0
    return np.concatenate(ts), np.concatenate(vs), np.array(starts)


def find_pulses(t, v):
    """(t0, indices of the flat top) for every complete pulse in the trace."""
    v_max = np.nanmax(v)
    on = v > 0.5 * v_max
    rises = np.flatnonzero(~on[:-1] & on[1:])
    falls = np.flatnonzero(on[:-1] & ~on[1:])
    pulses = []
    for rise in rises:
        later = falls[falls > rise]
        # Skip pulses cut off by the start or end of a file
        if np.isnan(v[rise]) or not len(later) or np.isnan(v[later[0] + 1]):
            continue
        idx = np.arange(rise + 1, later[0] + 1)
        # Noise on the decay tail after a pulse can cross the threshold too
        if np.median(v[idx]) < 0.8 * v_max:
            continue
        level = np.median(v[idx])
        half = (np.nanmedian(v[max(rise - 5, 0):rise + 1]) + level) / 2
        j = rise + 1
        while v[j - 1] >= half:
            j -= 1
        t0 = t[j - 1] + (half - v[j - 1]) / (v[j] - v[j - 1]) * (t[j] - t[j - 1])
        # Samples still on the rising edge are not part of the flat top
        below = v[idx] < PLATEAU_FRACTION * level
        lead = np.argmax(~below)
        idx, below = idx[lead:], below[lead:]
        # The flat top ends at the first sample well below the pulse's level
        if below.any():
            idx = idx[:np.argmax(below)]
        pulses.append((t0, idx))
    return pulses


def fit_sqrt_t(x, y):
    """Least squares of y = V0 + b x; returns V0, b, their standard errors."""
    design = np.column_stack([np.ones_like(x), x])
    p, res, *_ = np.linalg.lstsq(design, y, rcond=None)
    sigma2 = res[0] / (len(x) - 2)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return p[0], p[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def fit_samples(t, v, t0, idx, skip):
    """sqrt(t - t0) in sqrt(ms) and Vf of the samples of one pulse that go in the fit."""
    idx = idx[skip:-1]
    return np.sqrt((t[idx] - t0) * 1e3), v[idx]


def fit_pulse(t, v, t0, idx, skip):
    return fit_sqrt_t(*fit_samples(t, v, t0, idx, skip))


def average_pulse(t, v, pulses, dt):
    """Time from t0 (s) and Vf of the pulses averaged sample by sample, as in Fig. 4.

    Samples are binned on a grid of the sample interval from each pulse's t0, so every
    pulse puts one sample in each bin; bins that fewer than half the pulses reach (at the
    very start and end of the flat top) are dropped.
    """
    n_bins = max(int((t[idx[-1]] - t0) / dt) + 1 for t0, idx in pulses)
    t_sum, v_sum, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for t0, idx in pulses:
        k = ((t[idx] - t0) / dt).astype(int)
        np.add.at(t_sum, k, t[idx] - t0)
        np.add.at(v_sum, k, v[idx])
        np.add.at(count, k, 1)
    keep = count >= len(pulses) / 2
    return t_sum[keep] / count[keep], v_sum[keep] / count[keep]


def fit_average(t_avg, v_avg, t_skip):
    """sqrt(t) fit of the averaged pulse after t_skip (s), leaving out its last sample."""
    in_fit = t_avg >= t_skip
    in_fit[-1] = False
    return fit_sqrt_t(np.sqrt(t_avg[in_fit] * 1e3), v_avg[in_fit]), in_fit


def fit_window(t_avg, v_avg, window):
    """sqrt(t) fit of the averaged pulse over window = (lo, hi) in sqrt(ms)."""
    x = np.sqrt(t_avg * 1e3)
    in_window = (x >= window[0]) & (x <= window[1])
    return fit_sqrt_t(x[in_window], v_avg[in_window])


def plot_pulses(t_avg, v_avg, avg_fit, in_fit, n_pulses, fits, v0_mean, v0_err, skip,
                title, path, early=None):
    """Fig. 4 style averaged pulse against sqrt(t) with its fit, and Vf(0) of each pulse.

    early is an optional (fit, error, window) of the early part of the averaged pulse.
    """
    fig, (ax, ax_v0) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    x_avg = np.sqrt(t_avg * 1e3)
    ax.scatter(x_avg[in_fit], v_avg[in_fit], s=6, color=BLUE, zorder=3,
               label=f"Average of {n_pulses} pulses")
    ax.scatter(x_avg[~in_fit], v_avg[~in_fit], s=10, facecolors="none", edgecolors=MUTED,
               zorder=3, label=f"Not in fit (first {skip} samples, last one)")
    v0, b, v0_fit_err, _ = avg_fit
    x = np.linspace(0, x_avg[-1], 50)
    fit_name = "Whole pulse" if early else "Back extrapolated"
    ax.plot(x, v0 + b * x, color=INK, lw=1.5, zorder=4,
            label=f"{fit_name}, $V_f(0)$ = {v0:.4f} ± {v0_fit_err:.4f} V")
    if early:
        (v0_e, b_e, *_), v0_e_err, window = early
        x = np.linspace(0, window[1], 50)
        ax.axvspan(*window, color=ORANGE, alpha=0.1, lw=0, zorder=1)
        ax.plot(x, v0_e + b_e * x, color=ORANGE, lw=1.8, zorder=5,
                label=f"Early part ({window[0]:g}–{window[1]:g} √ms), "
                      f"$V_f(0)$ = {v0_e:.4f} ± {v0_e_err:.4f} V")
    ax.set_xlim(left=0)
    ax.set_xlabel(r"Square root of time / $\sqrt{\mathrm{ms}}$")
    ax.set_ylabel("Forward voltage / V")
    ax.set_title(f"{title}: average of {n_pulses} pulses", color=INK)
    ax.legend(frameon=False, loc="lower left")
    style(ax)

    n = np.arange(1, len(fits) + 1)

    ax_v0.errorbar(n, [f[0] for f in fits], yerr=[f[2] for f in fits],
                   fmt="o", color=BLUE, ms=4, capsize=2, zorder=3)
    ax_v0.axhline(v0_mean, color=ORANGE, lw=1.5, zorder=2, label="Mean")
    ax_v0.axhspan(v0_mean - v0_err, v0_mean + v0_err, color=ORANGE, alpha=0.15, lw=0)
    ax_v0.set_xlabel("Pulse")
    ax_v0.set_ylabel("$V_f(0)$ / V")
    ax_v0.set_title("Extrapolated forward voltage per pulse", color=INK)
    ax_v0.legend(frameon=False)
    style(ax_v0)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def plot_single_pulse(t, v, t0, idx, fit, skip, number, path):
    """One pulse against time, with the samples used in the fit and the fitted line."""
    window = (t > t0 - 0.02) & (t < t0 + 0.2)
    used = idx[skip:-1]
    skipped = np.r_[idx[:skip], idx[-1]]
    v0, b = fit[0], fit[1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot((t[window] - t0) * 1e3, v[window], color=FAINT, lw=1, zorder=2)
    ax.scatter((t[used] - t0) * 1e3, v[used], s=12, color=BLUE, zorder=3,
               label="Samples in fit")
    ax.scatter((t[skipped] - t0) * 1e3, v[skipped], s=20, facecolors="none",
               edgecolors=ORANGE, zorder=3, label="Samples skipped")
    t_line = np.linspace(0, (t[idx[-1]] - t0) * 1e3, 200)
    ax.plot(t_line, v0 + b * np.sqrt(t_line), color=INK, lw=1.5, zorder=4,
            label=rf"Fit $V_0 + b\sqrt{{t}}$, $V_f(0)$ = {v0:.4f} V")
    ax.axvline(0, color=MUTED, ls="--", lw=1, label="t = 0 (pulse start)")
    ax.set_xlabel("Time from pulse start / ms")
    ax.set_ylabel("Forward voltage / V")
    ax.set_title(f"Pulse {number}", color=INK)
    ax.legend(frameon=False, loc="lower center")
    style(ax)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def analyse(prefix, files, skip=SKIP_SAMPLES, alt_skips=ALT_SKIP_SAMPLES, early_window=None,
            alt_early_windows=()):
    t, v, file_starts = load_traces([DATA_DIR / f for f in files])
    pulses = find_pulses(t, v)
    dt = np.median(np.diff(t[np.isfinite(v)]))
    print(f"\n===== {', '.join(files)} =====")
    print(f"{len(pulses)} pulses, sample interval {dt * 1e3:.2f} ms, "
          f"{len(pulses[0][1])} samples on the flat top of the first")

    fits = [fit_pulse(t, v, t0, idx, skip) for t0, idx in pulses]
    print(f"\nVf = V0 + b sqrt(t), first {skip} samples ({skip * dt * 1e3:.2f} ms) skipped")
    print(" pulse   file          t0 (s)    V0 (V)            b (mV/sqrt(ms))   plateau mean (V)")
    for k, ((t0, idx), (v0, b, v0_err, b_err)) in enumerate(zip(pulses, fits), 1):
        f = np.searchsorted(file_starts, t0) - 1
        print(f"  {k:2d}   {files[f]:<12}  {t0 - file_starts[f]:7.3f}   {v0:.4f} ± {v0_err:.4f}   "
              f"{b * 1e3:6.2f} ± {b_err * 1e3:5.2f}     {v[idx].mean():.4f}")

    v0 = np.array([f[0] for f in fits])
    v0_mean, v0_sem = v0.mean(), v0.std(ddof=1) / np.sqrt(len(v0))

    # Systematic error: spread of the mean V0 over the alternative numbers of skipped samples
    alt = [np.mean([fit_pulse(t, v, t0, idx, s)[0] for t0, idx in pulses])
           for s in alt_skips]
    v0_sys = (max(alt + [v0_mean]) - min(alt + [v0_mean])) / 2
    v0_err = np.hypot(v0_sem, v0_sys)

    print("\nMean V0 against samples skipped:")
    for s, a in sorted(zip((skip, *alt_skips), [v0_mean, *alt])):
        print(f"  {s}: {a:.4f} V")
    print(f"\nVf(t = 0), mean of {len(v0)} pulses: {v0_mean:.4f} V"
          f"  ± {v0_sem:.4f} (stat) ± {v0_sys:.4f} (skip choice) = ± {v0_err:.4f} V"
          f"   (spread between pulses: {v0.std(ddof=1):.4f} V)")

    t_avg, v_avg = average_pulse(t, v, pulses, dt)
    t_skip = np.mean([t[idx[skip]] - t0 for t0, idx in pulses])
    avg_fit, in_fit = fit_average(t_avg, v_avg, t_skip)
    v0_avg, b_avg = avg_fit[0], avg_fit[1]
    drop = -b_avg * np.sqrt(t_avg[in_fit][-1] * 1e3)
    print(f"Fit to the averaged pulse: V0 = {v0_avg:.4f} ± {avg_fit[2]:.4f} V, "
          f"b = {b_avg * 1e3:.2f} ± {avg_fit[3] * 1e3:.2f} mV/sqrt(ms), "
          f"drop from t = 0 to the end of the pulse {drop * 1e3:.1f} mV "
          f"({drop / v0_avg * 100:.2f}%)")
    print(f"Plateau mean of all pulses (no extrapolation): "
          f"{np.mean([v[idx].mean() for _, idx in pulses]):.4f} V")

    early = None
    if early_window:
        early_fit = fit_window(t_avg, v_avg, early_window)
        # Systematic error: half the spread of V0 over the alternative windows
        alt_v0 = [fit_window(t_avg, v_avg, w)[0] for w in alt_early_windows]
        sys = (max(alt_v0 + [early_fit[0]]) - min(alt_v0 + [early_fit[0]])) / 2
        err = np.hypot(early_fit[2], sys)
        drop = early_fit[0] - v_avg[in_fit][-20:].mean()
        print(f"\nFit to the early part of the averaged pulse, sqrt(t) = "
              f"{early_window[0]:g}-{early_window[1]:g} sqrt(ms):")
        for w, a in zip((early_window, *alt_early_windows), (early_fit[0], *alt_v0)):
            print(f"  window {w[0]:g}-{w[1]:g}: V0 = {a:.4f} V")
        print(f"V0 = {early_fit[0]:.4f} ± {early_fit[2]:.4f} (stat) ± {sys:.4f} (window) "
              f"= ± {err:.4f} V, b = {early_fit[1] * 1e3:.1f} ± {early_fit[3] * 1e3:.1f} "
              f"mV/sqrt(ms), drop from t = 0 to the end of the pulse {drop * 1e3:.1f} mV "
              f"({drop / early_fit[0] * 100:.2f}%)")
        early = (early_fit, err, early_window)

    title = Path(files[0]).stem if len(files) == 1 else \
        f"{Path(files[0]).stem}–{Path(files[-1]).stem}"
    plot_pulses(t_avg, v_avg, avg_fit, in_fit, len(pulses), fits, v0_mean, v0_err, skip,
                title, OUTPUT_DIR / f"{prefix}.png", early)
    plot_single_pulse(t, v, *pulses[0], fits[0], skip, 1,
                      OUTPUT_DIR / f"{prefix}_pulse_1.png")


def main():
    for prefix, settings in DATASETS.items():
        analyse(prefix, **settings)


if __name__ == "__main__":
    main()
