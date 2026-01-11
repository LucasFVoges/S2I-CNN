import os.path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pass

def data_gen(n = 100, length = 10, resolution = 0.4, noise = 1, y_shift = 0.02, num_peaks = 1, peak_randomised_amount = 0.05, seed = 42):
    """
    generates a test Dataset
    Including 2 variants of a random spectrum (function).

    # TODO: ADD More Peaks to the second class
    # TODO: Length should not influence the peak parameter!

    :param n: number of spectra to generate for each Class! (default: 100).
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
    :param peak_randomised_amount: maximal amount added to the center, amplitude and widht of every peak (default: 0.05).
    :type peak_randomised_amount: float
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

    # Define the x-axis based on length and resolution
    num_points = int(length / resolution)
    x = np.linspace(0, length, num_points)
    spectra = []

    # Define the base spectrum parameters
    amplitudes = []
    centers = []
    widths = []
    for _ in range(num_peaks+1):                                # This can be directly done with np.random.uniform(0,1,num_peaks)
        amplitudes.append(np.random.uniform(0.1,1))
        centers.append(np.random.uniform(0,1) * length)
        widths.append(np.random.uniform(0.2,1) * (length * 0.1) + 0.1)

    for spec in range(n*2):
        # Create a base spectrum as a sum of min and max peaks random Gaussians
        y = np.ones_like(x)
        y = y + np.random.uniform(-y_shift, y_shift)

        # Adds random Gaussian peaks to base spectrum
        range_peaks = num_peaks if spec < n else num_peaks + 1
        for i in range(range_peaks):
            this_amplitude = amplitudes[i] + np.random.uniform(-peak_randomised_amount,peak_randomised_amount)
            this_center = centers[i] + np.random.uniform(-peak_randomised_amount,peak_randomised_amount)
            this_width = widths[i] + np.random.uniform(-peak_randomised_amount,peak_randomised_amount)
            y += this_amplitude * np.exp(-((x - this_center) ** 2) / (2 * this_width ** 2))

        # Add Gaussian noise
        y += np.random.normal(0, noise/100, size=x.shape)

        spectra.append(y)

    # Return as a DataFrame where each row is a spectrum
    df = pd.DataFrame(spectra)
    df.columns = x.round(3)

    return df
