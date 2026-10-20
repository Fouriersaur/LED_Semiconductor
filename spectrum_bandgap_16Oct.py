"""Bandgap from the LED emission spectrum, 16 Oct session (76.7-268.5 K).

Same analysis as spectrum_bandgap.py, on the CCS200 exports in data/LED_spectra_16Oct,
named 161026_<T>_pm<dT>.csv (161026_199_3_pm0_2 = 199.3 ± 0.2 K).

The spectrometer background background_1610_1.csv is subtracted from every spectrum. It was
taken at 0.1 s integration against 1.25 s for the spectra, but is not rescaled: the CCS200
reports intensity as a fraction of ADC full scale and the background is the readout offset,
which does not grow with integration time. Any offset left over is removed by the
off-peak median in load_spectrum.

The peak moves by ~60 meV and broadens from ~21 to ~39 meV over this range, so the fit
windows are set relative to each spectrum's peak instead of fixed in energy.
"""

from pathlib import Path

from spectrum_bandgap import run

SPECTRA_DIR = Path(__file__).with_name("data") / "LED_spectra_16Oct"
PLOT_PREFIX = "16_Oct_spectrum_LED"
BACKGROUND = SPECTRA_DIR / "background_1610_1.csv"

# Fit window (eV from the peak), and alternative windows used for the systematic error on Eg
FIT_WINDOW_EV = (-0.12, 0.15)
ALT_FIT_WINDOWS_EV = [(-0.08, 0.10), (-0.06, 0.08), (-0.16, 0.20)]
PLOT_WINDOW_EV = (-0.16, 0.20)


def main():
    run(SPECTRA_DIR, "161026_*.csv", PLOT_PREFIX,
        fit_ranges=lambda e_peak: [(e_peak + lo, e_peak + hi)
                                   for lo, hi in [FIT_WINDOW_EV, *ALT_FIT_WINDOWS_EV]],
        plot_range=lambda e_peak: (e_peak + PLOT_WINDOW_EV[0], e_peak + PLOT_WINDOW_EV[1]),
        overlay_xlims=((1.93, 2.10), (1.90, 2.15)), background=BACKGROUND)


if __name__ == "__main__":
    main()
