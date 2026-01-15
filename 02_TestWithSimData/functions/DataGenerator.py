import os.path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pass

def data_gen(n = 10,
             length = 50,
             resolution = 0.4,
             noise = 1,
             y_shift = 0.02,
             num_peaks = 1,
             num_peaks_addition = 1,
             peak_randomised_amount = 0.05,
             normalize = True,
             normalize_individually = False,
             seed = 42):
    """
    generates a test Dataset
    Including 2 variants of a random spectrum (added normal distributions).
    Spectra can be normalized to [0,1].

    :param n: number of spectra to generate for each Class! (default: 10).
    :type n: int
    :param length: number of spectrapoints to generate (default: 10, min: 10).
    :type length: int
    :param resolution: The resolution of the spectra (default: 0.4).
    :type resolution: float
    :param noise: The noise level (default: 1).
    :type noise: float
    :param y_shift: max amount of y value to be shifted (default: 0.02).
    :type y_shift: float
    :param num_peaks: number of peaks (default: 1, min: 1 cannot be larger than length/2).
    :type num_peaks: int
    :param num_peaks_addition: Number of peaks that are added to the spectrum in the second class (1 or more).
    :type num_peaks_addition: int
    :param peak_randomised_amount: maximal amount added to the center, amplitude and widht of every peak (default: 0.05).
    :type peak_randomised_amount: float
    :param normalize: If data should be normalized to [0,1] (default: False).
    :type normalize: bool
    :param normalize_individually: If the normalization should be done individually for each spectrum. Else the whole dataset is scaled betweet 0 and 1 (default: True).
    :type normalize_individually: bool
    :param seed: random seed for reproducibility (default: 42).
    :type seed: int
    :return: data frame with spectral intensities
    :rtype: pd data frame
    """

    np.random.seed(seed)

    # checks:
    if length < 10: length = 10
    if n < 1: n = 1
    if resolution > length: resolution = length
    if num_peaks < 1: num_peaks = 1
    if num_peaks > length/2: num_peaks = length/2
    if num_peaks_addition < 1: num_peaks_addition = 1
    if noise <= 0: noise = 0.001

    # Define the x-axis based on length and resolution
    num_points = int(length / resolution)
    x = np.linspace(0, length, num_points)
    spectra = []

    # Define the base spectrum parameters
    amplitudes = np.random.uniform(0.1,1,num_peaks+num_peaks_addition)
    centers = np.random.uniform(0,1,num_peaks+num_peaks_addition) * length
    widths = np.random.uniform(1,100,num_peaks+num_peaks_addition)

    for spec in range(n*2):
        # Create a base spectrum as a sum of min and max peaks random Gaussians
        y = np.ones_like(x)
        y = y + np.random.uniform(-y_shift, y_shift)


        range_peaks = num_peaks if spec < n else num_peaks + num_peaks_addition # Only Add addition Peaks for half of the data
        # Adds random Gaussian peaks to base spectrum
        for i in range(range_peaks):
            this_amplitude = amplitudes[i] + np.random.uniform(0,peak_randomised_amount)
            this_center = centers[i] + np.random.uniform(-peak_randomised_amount*10,peak_randomised_amount*10)
            this_width = widths[i] + np.random.uniform(0,peak_randomised_amount*50)
            y += this_amplitude * np.exp(-((x - this_center) ** 2) / (2 * this_width ** 2))

        # Add Gaussian noise
        y += np.random.normal(0, noise/100, size=x.shape)

        spectra.append(y)

    # Normalize if needed
    if normalize:
        if normalize_individually:
            spectra = [(s - np.min(s)) / (np.max(s) - np.min(s)) for s in spectra]
        else:
            spectra_array = np.array(spectra)
            spectra = (spectra_array - np.min(spectra_array)) / (np.max(spectra_array) - np.min(spectra_array))

    # Return as a DataFrame where each row is a spectrum
    df = pd.DataFrame(spectra)
    df.columns = x.round(3)

    return df
