from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import numpy as np
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig

from matplotlib.backend_bases import MouseEvent, KeyEvent

MWSPEC_DEBUG = False

space_delim = ["ft"]
comma_delim = ["csv"]

plot_spectrum_flag = False
plot_RVI_flag = False
plot_ratio_track = False

fig = plt.figure()
ax = fig.add_subplot()

clicked_points = []

cdp_click_thresh = 0.005

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

    # EXPENSIVE
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

    def remove_peaks_from(self, other: Union[SpectrumPeaks, set[SpectrumPeaks]], freq_var: float) -> Union[(np.array, np.array), dict[SpectrumPeaks, (np.array, np.array)]]:
        if isinstance(other, set):
            output = {}
            for spectrum in other:
                iself, iother = self.correlate_peaks(spectrum, freq_var)
                self.remove_peaks(iself)
                output[spectrum] = iself, iother
            return output
        else:
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

        global plot_spectrum_flag
        plot_spectrum_flag = True

class ExperimentalSpectrum(SpectrumPeaks):
    blank_value = 0

    peak_min_inten = None
    peak_min_prominence = None
    peak_wlen = None

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
        self.ipeak_left = np.delete(self.ipeak_left, inds)
        self.ipeak_right = np.delete(self.ipeak_right, inds)

    @staticmethod
    @njit(cache=True)
    def _remove_spec_peaks(inds: np.array, inten: np.array, lefts: np.array, rights: np.array, blank_val: float) -> None:
        for i in inds:
            inten[lefts[i]:rights[i]] = blank_val

    def plot(self, label=None):
        label = self.name if label is None else label
        plt.plot(self.freq, self.inten, label=label)

        global plot_spectrum_flag
        plot_spectrum_flag = True

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

    # Returns: ratios, iself, iother
    def divide_by(self, other: ExperimentalSpectrum, freq_var: float, export_filename: str = None) -> (np.array, np.array, np.array):
        iself, iother = self.correlate_peaks(other, freq_var)

        ratios = self.peak_intens()[iself] / other.peak_intens()[iother]

        if export_filename:
            stacked = np.column_stack([self.peak_freqs()[iself], other.peak_freqs()[iother], ratios])
            np.savetxt(export_filename, stacked, header=f"{self.name},{other.name},Ratios", delimiter=",")

        return ratios, iself, iother
    
    def keep_ratios_of(self, other: ExperimentalSpectrum, freq_var: float, lower_ratio: float, upper_ratio: float, apply_to_other: bool = False) -> None:
        ratios, iself, iother = self.divide_by(other, freq_var)

        ratio_mask = (ratios >= lower_ratio) & (ratios <= upper_ratio)

        iself = iself[ratio_mask]

        iself_mask = np.arange(len(self.peak_freqs()))
        iself_mask = np.delete(iself_mask, iself)

        self.remove_peaks(iself_mask)

        if apply_to_other:
            iother = iother[ratio_mask]
            iother_mask = np.arange(len(other.peak_freqs()))
            iother_mask = np.delete(iother_mask, iother)

            other.remove_peaks(iother_mask)

        # Returns: iself, iother, ifind, ratios
    def obtain_ratios_of(self, other: ExperimentalSpectrum, to_find: SpectrumPeaks, freq_var) -> (np.array, np.array):
        ratios, iself_ratio, iother_ratio = self.divide_by(other, freq_var)
        iself_corr, ifind_corr = self.correlate_peaks(to_find, freq_var)

        # The returned peaks must be both correlated and have a counterpart in the other spectrum
        _, iself_ratio_inds, iself_corr_inds = np.intersect1d(iself_ratio, iself_corr, return_indices=True)

        return iself_ratio[iself_ratio_inds], iother_ratio[iself_ratio_inds], ifind_corr[iself_corr_inds], ratios[iself_ratio_inds]

    # EXPENSIVE
    def find_CDPS(self, freq_var: float, max_double_step: float, max_cdp_step, max_inten_var: float) -> None:
        cdps = ExperimentalSpectrum._find_CDPS(self.peak_freqs(), self.peak_intens(), freq_var, max_double_step, max_cdp_step, max_inten_var)

        mask = np.arange(len(self.peak_freqs()))
        mask = np.delete(mask, cdps)

        self.remove_peaks(mask)

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
    
    def find_intensity_of(self, other: SpectrumPeaks, freq_var: float) -> (np.array, np.array):
        iself, iother = self.correlate_peaks(other, freq_var)
        return iself, self.peak_intens[iself]

    def cut_and_replace_plot(self, others: set[SpectrumPeaks], freq_var: float) -> None:
        color_counter = 1
        for other in others:
            iself, iother = self.correlate_peaks(other, freq_var)

            plt.stem(self.peak_freqs()[iself], self.peak_intens()[iself], label=other.name, markerfmt=" ", linefmt="C" + str(color_counter))
            color_counter += 1

        self.remove_peaks_from(others, freq_var)
        self.plot()
    
class GeneratedSpectrum(SpectrumPeaks):
    freq: np.array
    inten: np.array
    uJ: np.array
    uKa: np.array
    uKc: np.array
    lJ: np.array
    lKa: np.array
    lKc: np.array

    def peak_freqs(self):
        return self.freq
    
    def peak_intens(self):
        return self.inten
    
    def remove_peaks(self, inds):
        self.freq = np.delete(self.freq, inds)
        self.inten = np.delete(self.inten, inds)

class FrequencyList(SpectrumPeaks):
    freq: np.array

    def peak_freqs(self):
        return self.freq
    
    def peak_intens(self):  # TODO: Find alternative to making this SpectrumPeaks
        raise ValueError(f"Object with name \"{self.name}\" is a FrequencyList and doesn't contain peak intensities")
    
    def remove_peaks(self, inds):
        self.freq = np.delete(self.freq, inds)

def get_spectrum(filename: str, name: str, peak_min_inten: float = None, peak_min_prominence: float = None, peak_wlen: int = None, skiprows=0) -> ExperimentalSpectrum:
    split_name, ext = _check_filename(filename)

    name = name if name is not None else split_name

    if ext in space_delim:
        data = np.loadtxt(filename, delimiter=" ", skiprows=skiprows)
    elif ext in comma_delim:
        data = np.loadtxt(filename, delimiter=",", skiprows=skiprows)
    else:
        raise ValueError(f"The spectrum file \"{filename}\" has an unsupported extension")

    freq, inten = data[:, 0], data[:, 1]

    peak_min_inten = peak_min_inten if peak_min_inten is not None else ExperimentalSpectrum.peak_min_inten
    peak_min_prominence = peak_min_prominence if peak_min_prominence is not None else ExperimentalSpectrum.peak_min_prominence
    peak_wlen = peak_wlen if peak_wlen is not None else ExperimentalSpectrum.peak_wlen

    peaks, properties = sig.find_peaks(inten, height=peak_min_inten, prominence=peak_min_prominence, wlen=peak_wlen)

    spec = ExperimentalSpectrum()
    spec.name = name.split("/")[-1]
    spec.freq = freq
    spec.inten = inten
    spec.ipeaks = peaks
    spec.ipeak_left = properties["left_bases"]
    spec.ipeak_right = properties["right_bases"]
    debug(f"Loaded ExperimentalSpectrum with name {name} with {len(peaks)} peaks discovered")

    return spec

def get_lin(filename: str, name: str = None ):
    split_name, ext = _check_filename(filename)

    name = split_name if name is None else name

    if ext != "lin":
        raise ValueError(f"File \"{filename}\" does not have a .lin extension")
    
    data = np.loadtxt(filename)
    
    spec = GeneratedSpectrum()
    spec.name = name.split("/")[-1]
    spec.freq = data[:, 12]
    spec.inten = np.zeros(len(spec.freq))  # TODO: Figure out a better way to deal with intensity
    spec.uJ = data[:, 0].astype("int")
    spec.uKa = data[:, 1].astype("int")
    spec.uKc = data[:, 2].astype("int")
    spec.lJ = data[:, 3].astype("int")
    spec.lKa = data[:, 4].astype("int")
    spec.lKc = data[:, 5].astype("int")

    return spec

def get_cat(filename: str, name: str = None):
    split_name, ext = _check_filename(filename)

    name = split_name if name is None else name

    if ext != "cat":
        raise ValueError(f"File \"{filename}\" does not have a .lin extension")
    
    data = np.genfromtxt(filename, delimiter=[13, 8, 8, 2, 10, 3, 7, 4, 2, 2, 2, 8, 2, 2])

    spec = GeneratedSpectrum()
    spec.name = name.split("/")[-1]
    spec.freq = data[:, 0]
    spec.inten = np.pow(10, data[:, 2])
    spec.uJ = data[:, 8].astype("int")
    spec.uKa = data[:, 9].astype("int")
    spec.uKc = data[:, 10].astype("int")
    spec.lJ = data[:, 11].astype("int")
    spec.lKa = data[:, 12].astype("int")
    spec.lKc = data[:, 13].astype("int")

    return spec

def get_line_list(filename: str, name: str = None) -> FrequencyList:
    split_name, ext = _check_filename(filename)

    name = name if name is not None else split_name
    
    data = np.loadtxt(filename)

    spec = FrequencyList()
    spec.name = name.split("/")[-1]
    spec.freq = data

    return spec


def show(inten_units=None) -> None:
    plt.legend(loc="upper left")
    if plot_spectrum_flag and plot_RVI_flag:
        raise ValueError("Spectra and ratio plots cannot be plotted together!")
    elif plot_spectrum_flag:
        inten_units = "a.u." if inten_units is None else inten_units
        plt.xlabel("Frequency (MHz)")
        plt.ylabel(f"Intensity ({inten_units})")
        plt.show()
    elif plot_ratio_track:
        plt.xlabel("Concentration Ratio")
        plt.ylabel("Intensity Ratio")
        plt.show()

def plot_RVI(inten_spec: ExperimentalSpectrum, divisor_spec: ExperimentalSpectrum, freq_var: float) -> None:
    ratios, iself, iother = inten_spec.divide_by(divisor_spec, freq_var)

    plt.scatter(inten_spec.peak_intens()[iself], ratios)

    global plot_RVI_flag
    plot_RVI_flag = True

def construct_RVI(inten_spec: ExperimentalSpectrum, divisor_spec: ExperimentalSpectrum, freq_var: float, others: set[SpectrumPeaks] = None) -> None:
    if others is not None:
        for other in others:
            iinten, idiv, iother, ratios = inten_spec.obtain_ratios_of(divisor_spec, other, freq_var)

            plt.scatter(inten_spec.peak_intens()[iinten], ratios, label=other.name)

        inten_spec.remove_peaks_from(others, freq_var)

    ratios, iself, iother = inten_spec.divide_by(divisor_spec, freq_var)

    plt.scatter(inten_spec.peak_intens()[iself], ratios, c='black')

    plt.title(f"{inten_spec.name} / {divisor_spec.name}")
    plt.xlabel(f"Intensity in {inten_spec.name}")
    plt.ylabel(f"Ratio of {inten_spec.name} / {divisor_spec.name}")
    plt.show()
    

def plot_ratio_boxes(base_spec: ExperimentalSpectrum, divisor_spec: ExperimentalSpectrum, to_find: list[SpectrumPeaks], freq_var: float) -> None:
    ratio_list = []
    labels = []

    for find_spec in to_find:
        iself, iother, ifind, ratios = base_spec.obtain_ratios_of(other=divisor_spec, to_find=find_spec, freq_var=freq_var)
        ratio_list.append(ratios)
        labels.append(find_spec.name)

    plt.boxplot(x=ratio_list, labels=labels)
    
    global plot_RVI_flag
    plot_RVI_flag = True

def plot_ratio_track(base: ExperimentalSpectrum, divisors: list[ExperimentalSpectrum], species: SpectrumPeaks, freq_var: float) -> None:
    for divisor in divisors:
        iself, iother, ifind, ratios = base.obtain_ratios_of(divisor, species, freq_var)

        plt.scatter(np.full(len(ratios), fill_value))

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

def on_click(event: KeyEvent):
    if event.inaxes is not None and event.key == " ":
        print(f"Clicked at {event.xdata}, {event.ydata}")

        global clicked_points
        clicked_points.append(event.xdata)

        if len(clicked_points) == 4:
            first_diff = clicked_points[1] - clicked_points[0]
            second_diff =clicked_points[2] - clicked_points[3]
            result = abs(first_diff - second_diff)
            if result < cdp_click_thresh:
                print(f"Differences are {first_diff} and {second_diff}, is a confirmed CDP")
            else:
                print(f"Differences are {first_diff} and {second_diff}, not a CDP")
            clicked_points.clear()

cid = fig.canvas.mpl_connect('key_press_event', on_click)