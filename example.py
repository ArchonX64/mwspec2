from mwspec2 import *

activate_debug()  # This will log important events in the console, can be nice to check if things are happening correctly

# ----------------------------------------------------------------------------------------------------------------------------------------------
# LOADING IN DATA

# These are the various values needed to determine what counts as a 'peak' in an experimentally collected spectrum
# These values will apply to every spectrum loaded through 'get_spectrum'
# If these value differ between spectra, they can be put in as arguments in 'get_spectrum'
# Their meanings are explained in the documentation
ExperimentalSpectrum.peak_min_inten=30
ExperimentalSpectrum.peak_min_prominence=10
ExperimentalSpectrum.peak_wlen=20

# Load an experimental spectrum from a file into a variable
# Must in the format "Frequency  Intensity" or "Frequency,Intensity"
# Optional name can be given as well as any rows to skip at the start
spectrum1 = get_spectrum("spectrum1.ft", name="Spectrum1", skiprows=1)
spectrum2 = get_spectrum("spectrum2.csv", name="Spectrum2", skiprows=1, peak_min_inten=1)  # The peak variables can individually listed here

# Generated spectra/linelists can also be loaded, by using get_lin, get_cat, or get_line_list depending on the 
generated1 = get_lin("linfile.lin")  # .lin file from SPFIT
generated2 = get_cat("catfile.cat")  # .cat file from SPCAT
generated3 = get_line_list("linelist.txt")  # A newline delimited list of frequencies

# -----------------------------------------------------------------------------------------------------------------------------------------------
# ACCESSING PEAKS

# Many of these functions will return indexes of peaks, which have no numerical data of their own.
# They can be put back into their equivalent spectrum though to receive their data

spectrum1_indexes, generated2_indexes = spectrum1.some_random_function(other=generated2)  # Common output result

spectrum1_frequencies = spectrum1.peak_freqs()[spectrum1_indexes]  # Will return a list of corresponding frequencies given by the list

spectrum1_intensities = spectrum1.peak_intens()[spectrum1_indexes]  # Same, but now for intensities

generated2_uJ = generated2.uJ[generated2_indexes]  # Generated spectra can have the quantum numbers accessed to, in this case the upper J value

# -----------------------------------------------------------------------------------------------------------------------------------------------
# PLOTTING DATA

# Each spectrum variable has a 'plot' function.

spectrum1.plot()
spectrum2.plot(label="A custom label")

spectrum1.plot_peaks(scatter=True, sides=False)  # The generated peaks of an experimental spectrum can be plotted too.
                                                 # 'Sides' will show the computed edges of each peak which will be used in cutting.

show()  # This command will halt execution and display everything that includes 'plot up until this point

# -----------------------------------------------------------------------------------------------------------------------------------------------
# CUTTING TRANSITIONS FROM AN EXPERIMENTAL SPECTRUM

# Lines from any type of spectrum can be removed from an experimental spectrum
# They will be replaced with a flat line of intensity 0, which makes it obvious when oberving

# Cutting requires: A set of all the frequencies you want removed (a list of spectra whose peaks to remove)
# A maximum frequency variability, which details the maximum difference in frequency (in MHz!!) two peaks between different
#   spectra can have to be considered the same peak to be cut
# This will cut the spectrum IN-PLACE

results = spectrum1.remove_peaks_from(other={generated1, generated2, generated3}, freq_var=0.05)

# The output will be two lists, each of them the indexes of cut peaks of the spectrum being modified (self_indexes -> spectrum1)
#   The other will be the indexes of the peaks being cut (other_indexes -> spectra in 'other')
# If multiple spectra are in 'other', then it will be a dictionary corresponding each spectrum object to its cut members

# -------------------------------------------------------------------------------------------------------------------------------------------------
# SPECTRUM DIVISION

# The intensities of transitions that are the same between two experimental spectra can be divided 
#   to produce a list of intensity ratios between peaks in the spectra
#   Requires another experimental spectrum, a maximum frequency correlation value, and optionally
#   a filename if you want to export the ratios.

ratios, self_indexes, other_indexes = spectrum1.divide_by(other=spectrum2, freq_var=0.05, export_filename="ratios.txt")

# Will return three lists, one containing the ratio values, the indexes of the numerator peaks (spectrum1),
#   and the indexes of the denominator (spectrum2)
#   These lists are index-matched, meaning for a given index n, ratio[n] will be the ratio of self_indexes[n] to other_indexes[n],
#   where the indexes are the indexes of their respective peak lists.

# -------------------------------------------------------------------------------------------------------------------------------------------------
# RATIO ELIMINATION

# This function will cut an experimental spectrum, so that only peaks that have a certain ratio intensity between 'other' will be kept
# other: Spectrum to divide by (denominator)
# freq_var: Maximum frequency difference between two peaks in different spectra to be considered the same
# lower_ratio: Lower end of kept ratios (must be < upper_ratio)
# upper_ratio: Upper end of kept ratios (must be > lower_ratio)
# apply_to_other: Normally, will only cut transitions out of spectrum it is called on (spectrum1). 
#   Making this true will cut it out of 'other' (spectrum2) too.

spectrum1.keep_ratios_of(other=spectrum2, freq_var=0.05, lower_ratio=1.2, upper_ratio=1.3, apply_to_other=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------
# OBTAINING THE RATIOS OF A CERTAIN SPECIES

# Finds the intensity ratios of a certain species within an experimental spectrum.
# denominator: The spectrum to divide by to obtain ratios
# to_find: The generated spectrum whose intensity ratios between the two experimental spectra will be found
# freq_var: Maximum frequency difference between two peaks in different spectra to be considered the same

spectrum1_indexes, spectrum2_indexes, generated1_indexes, ratios = spectrum1.obtain_ratios_of(denominator=spectrum2, to_find=generated1, freq_var=0.05)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# CREATING RVI PLOTS

# For creating ratio vs intensity plots, with the ability to remove peaks if needed
# inten_spec: The numerator spectrum and the one whose intensity is on the x-axis
# divisor_spec: The denominator spectrum
# freq_var: Maximum frequency difference between two peaks in different spectra to be considered the same
# others: Any frequency list of peaks who want to be identified. These points will be a different
#   color than the rest of the spectrum and 

construct_RVI(inten_spec=spectrum1, divisor_spec=spectrum2, freq_var=0.05, others={generated1, generated2})

# show() does NOT need to be called here

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# CONSTANT DIFFERENCE PATTERN FINDING

# This function will find the constant-difference patterns (CDP) within a spectrum, and cut a spectrum until only they remain.
# freq_var: The maximum frequency difference between two doublets for it to be considered a CDP
# max_double_step: The maximum frequency that two peaks can be seperated by to be considered part of a CDP
# max_cdp_step: The maximum frequency that two doublets can be sperated by to be considered a CDP
# max_inten_var: The maximum intensity ratio difference that the two 'similar' peaks in a CDP can have

spectrum1.find_CDPS(freq_var=0.05, max_double_step=100, max_cdp_step=20, max_inten_var=1.3)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 