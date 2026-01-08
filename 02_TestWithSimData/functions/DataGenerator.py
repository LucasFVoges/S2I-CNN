import os.path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pass

def data_gen(n = 100, length = 10, resolution = 0.4, noise = 1, num_peaks = 1,  seed = 42):
    """
    generates a test Dataset
    Including 2 variants of a random spectrum (function).


    :param n: number of spectra to generate (default: 100, < 2 will be set to 2).
    :type n: int
    :param length: number of spectrapoints to generate (default: 10, min: 10).
    :type length: int
    :param resolution: The resolution of the spectra (default: 0.4).
    :type resolution: float
    :param noise: The noise level (default: 1).
    :type noise: float
    :param num_peaks: number of peaks (default: 1, min: 1 cannot be larger than length/2).
    :type num_peaks: int
    :param seed: random seed for reproducibility (default: 42).
    :type seed: int
    :return: data frame with spectral intensities
    :rtype: pd data frame
    """

    np.random.seed(seed)

    # checks:
    if length < 10: length = 10
    if n < 2: n = 2
    if resolution > length: resolution = length
    if num_peaks < 1: num_peaks = 1
    if num_peaks > length/2: num_peaks = length/2

    # Define the x-axis based on length and resolution
    num_points = int(length / resolution)
    x = np.linspace(0, length, num_points)
    spectra = []

    for _ in range(n):
        # Create a base spectrum as a sum of min and max peaks random Gaussians
        y = np.ones_like(x)

        # Adds random Gaussian peaks to base spectrum
        for _ in range(num_peaks):
            amplitude = np.random.rand()
            center = np.random.rand() * length
            width = np.random.rand() * (length * 0.1) + 0.1
            y += amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        # Add Gaussian noise
        y += np.random.normal(0, noise/100, size=x.shape)

        spectra.append(y)

    # Return as a DataFrame where each row is a spectrum
    df = pd.DataFrame(spectra)
    df.columns = x.round(3)

    return df
