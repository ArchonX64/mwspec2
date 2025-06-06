from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig

MWSPEC_DEBUG = False

space_delim = ["ft"]
comma_delim = ["csv"]

plot_spectrum = False
plot_RVI = False

fig = plt.figure()
ax = fig.add_subplot()

class SpectrumPeaks:
    peak_color_index = 1  # Used to ensure each peak plot is a different color

    name: str
    color_index: int

    def __init__(self):
        self.color_index = SpectrumPeaks.peak_color_index
        SpectrumPeaks.peak_color_index += 1

    @abstractmethod
    def peak_freqs(self) -> np.array:
        pass

    @abstractmethod
    def peak_intens(self) -> np.array:
        pass

    @abstractmethod
    def remove_peaks(self, inds: np.array) -> None:
        pass

    def correlate_peaks(self, other: SpectrumPeaks, freq_var: float) -> (np.array, np.array):
        debug(f"Correlating {self.name} with {other.name}, which have {len(self.peak_freqs())} and {len(other.peak_freqs())} peaks respectively")
        iself, iother = SpectrumPeaks._correlate_peaks(self.peak_freqs(), other.peak_freqs(), freq_var)
        debug(f"Correlation complete with {len(iself)} correlations found")
        return iself, iother

    @staticmethod
    @njit(cache=True)
    def _correlate_peaks(self: np.array, other: np.array, fv: float) -> (np.array, np.array):
        self_result = np.zeros(len(self), dtype="int")
        other_result = np.zeros(len(other), dtype="int")
        counter = 0

        # Use searchsorted to find the closest values to each frequency
        iother_sort = np.searchsorted(self, other)

        for iother, iself in enumerate(iother_sort):
            # Edge Cases
            if iself == 0 or iself == len(self): continue

            # Determine the variation between possible correlations
            after_variance = self[iself] - other[iother]
            on_variance = other[iother] - self[iself - 1]

            # Determine if pair is within frequency variance
            # np.searchsorted put it between two values, so check both
            #   to see whether either are within the frequency variance
            if on_variance < fv and after_variance < fv:
                if on_variance < after_variance:
                    self_result[counter] = iself - 1
                    other_result[counter] = iother
                    counter += 1
                else:
                    self_result[counter] = iself
                    other_result[counter] = iother
                    counter += 1
            elif on_variance < fv:
                self_result[counter] = iself - 1
                other_result[counter] = iother
                counter += 1
            elif after_variance < fv:
                self_result[counter] = iself
                other_result[counter] = iother
                counter += 1

        # Trim unused space
        self_result = self_result[:counter]
        other_result = other_result[:counter]

        return self_result, other_result

    def remove_peaks_from(self, other: SpectrumPeaks, freq_var: float) -> (np.array, np.array):
        iself, iother = self.correlate_peaks(other, freq_var)
        self.remove_peaks(iself)
        return iself, iother

    def plot_peaks(self, scatter=False, name=None):
        name = self.name if name is None else name

        if not scatter:
            plt.stem(self.peak_freqs(), self.peak_intens(), label=name, markerfmt=" ",
                     linefmt="C" + str(self.color_index))
        else:
            plt.scatter(self.peak_freqs(), self.peak_intens(), label=name,
                     c="C" + str(self.color_index))

        global plot_spectrum
        plot_spectrum = True

class ExperimentalSpectrum(SpectrumPeaks):
    blank_value = 0

    freq: np.array
    inten: np.array

    ipeaks: np.array
    ipeak_left: np.array
    ipeak_right: np.array

    def peak_freqs(self) -> np.array:
        return self.freq[self.ipeaks]

    def peak_intens(self) -> np.array:
        return self.inten[self.ipeaks]

    def remove_peaks(self, inds: np.array) -> None:
        ExperimentalSpectrum._remove_spec_peaks(inds, self.inten, self.ipeak_left, self.ipeak_right,
                                                ExperimentalSpectrum.blank_value)
        self.ipeaks = np.delete(self.ipeaks, inds)
        self.ipeak_lefts = np.delete(self.ipeak_left, inds)
        self.ipeak_rights = np.delete(self.ipeak_right, inds)

    @staticmethod
    @njit(cache=True)
    def _remove_spec_peaks(inds: np.array, inten: np.array, lefts: np.array, rights: np.array, blank_val: float) -> None:
        for i in inds:
            inten[lefts[i]:rights[i]] = blank_val

    def plot(self):
        plt.plot(self.freq, self.inten)

        global plot_spectrum
        plot_spectrum = True

    def plot_peaks(self, scatter=False, name=None, sides=False):
        super().plot_peaks(scatter, name)

        name = self.name if name is None else name

        if sides:
            plt.scatter(self.freq[self.ipeak_left], self.inten[self.ipeak_left], label=name + "_left",
                        c="C" + str(self.color_index))
            plt.scatter(self.freq[self.ipeak_right], self.inten[self.ipeak_right], label=name + "_right",
                        c="C" + str(self.color_index))

    def export(self, filename: str) -> None:
        ext = _check_filename(filename)[1]

        data = np.column_stack([self.freq, self.inten])
        if ext in space_delim:
            np.savetxt(filename, data, delimiter=" ")
        elif ext in comma_delim:
            np.savetxt(filename, data, delimiter=",")

    def divide_by(self, other: ExperimentalSpectrum, freq_var: float) -> (np.array, np.array, np.array):
        iself, iother = self.correlate_peaks(other, freq_var)

        ratios = self.peak_intens()[iself] / other.peak_intens()[iother]

        return ratios, iself, iother
    
    def keep_ratios_of(self, other: ExperimentalSpectrum, freq_var: float, lower_ratio: float, upper_ratio: float) -> None:
        ratios, iself, iother = self.divide_by(other, freq_var)

        iself = iself[(ratios >= lower_ratio) & (ratios <= upper_ratio)]

        iself_mask = np.arange(len(self.peak_freqs()))
        iself_mask = np.delete(iself_mask, iself)

        self.remove_peaks(iself_mask)

    def find_CDPS(self, freq_var: float, max_double_step: float, max_cdp_step, max_inten_var: float) -> np.array:
        return ExperimentalSpectrum._find_CDPS(self.peak_freqs(), self.peak_intens(), freq_var, max_double_step, max_cdp_step, max_inten_var)

    @staticmethod
    @njit(cache=True)
    def _find_CDPS(freq: np.array, inten: np.array, freq_var: float,
                   max_double_step: float,max_cdp_step: float, max_inten_var: float) -> np.array:
        n = len(freq)

        # Find Differences
        doublets = np.full((n, n), -1.0)
        for i in range(n):
            for j in range(i + 1, n):
                diff = freq[j] - freq[i]
                if diff < max_double_step: doublets[i][j] = diff

        # Find CDPs
        cdps = []
        for left1 in range(n - 3):
            for left2 in range(left1 + 1, n - 2):
                for right1 in range(left2 + 1, n - 1):
                    for right2 in range(right1 + 1, n):
                        # Ignore values that were eliminated by not being in range
                        if doublets[left1][left2] < 0.0 or doublets[right1][right2] < 0:
                            break

                        # Check for CDP
                        if abs(doublets[left1][left2] - doublets[right1][right2]) < freq_var:
                            # Check for intensity ratio
                            if abs(inten[left1] / inten[right2] - 1) < max_inten_var and abs(inten[left2] / inten[right1] - 1) < max_inten_var:
                                cdps.append(left1)
                                cdps.append(left2)
                                cdps.append(right1)
                                cdps.append(right2)
                    if freq[right1] - freq[left2] > max_cdp_step:
                        break
        return cdps


def get_spectrum(filename: str, name: str, peak_min_inten: float, peak_min_prominence: float, peak_wlen: int)\
        -> ExperimentalSpectrum:
    name, ext = _check_filename(filename)

    if ext in space_delim:
        data = np.loadtxt(filename, delimiter=" ")
    elif ext in comma_delim:
        data = np.loadtxt(filename, delimiter=",")

    freq, inten = data[:, 0], data[:, 1]

    peaks, properties = sig.find_peaks(inten, height=peak_min_inten, prominence=peak_min_prominence, wlen=peak_wlen)

    spec = ExperimentalSpectrum()
    spec.name = name
    spec.freq = freq
    spec.inten = inten
    spec.ipeaks = peaks
    spec.ipeak_left = properties["left_bases"]
    spec.ipeak_right = properties["right_bases"]
    debug(f"Loaded ExperimentalSpectrum with name {name} with {len(peaks)} peaks discovered")

    return spec


def show(inten_units=None):
    plt.legend()
    if plot_spectrum and plot_RVI:
        raise ValueError("Spectra and ratio plots cannot be plotted together!")
    elif plot_spectrum:
        inten_units = "a.u." if inten_units is None else inten_units
        plt.xlabel("Frequency (MHz)")
        plt.ylabel(f"Intensity ({inten_units})")
        plt.show()


def _check_filename(filename: str):
    dotsplit = filename.split(".")
    if len(dotsplit) != 2:  # Verify that filename contains no extra "."
        raise ValueError(f"The filename {filename} cannot contain more than one \".\"!")
    return dotsplit


def activate_debug():
    global MWSPEC_DEBUG
    MWSPEC_DEBUG = True


def debug(string: str):
    time_str = formatted_time = datetime.now().strftime('%H:%M:%S') + f":{datetime.now().microsecond // 1000:03d}"
    if MWSPEC_DEBUG: print(f"[{time_str}] {string}")