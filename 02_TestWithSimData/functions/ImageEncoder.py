import os.path
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    pass

def image_encoder(data, data_keys = ["raw", "d1", "d2", "d3"], spectra_num = 0, shema='RGB'):
    """
    Converts spectral intensities to an Image. (1 Spectra)

    :param data: Spectral object as dictionary with keys: 'raw', 'd1', 'd2', 'd3'.
    :type data: dataframe with the 1st, 2nd and 3rd derivative.
    :param data_keys: List of keys in the dictionary.
    :type data_keys: list of str
    :param spectra_num: The number of the spectra in the dataframe.
    :type spectra_num: int
    :param shema: Color Shema 'BW', 'RGB', 'CMYK', 'HSV', 'LAB', 'L' (grayscale 2D) - (default: 'BW').
    :type shema: str
    :return: Value of success. Saves an image at the given output path.
    :rtype: bool
    """

    # checking input
    shema = shema.upper()
    if shema not in ['BW','RGB', 'CMYK', 'HSV', 'LAB', 'L']:
        print('Invalid color shema.')
        return False

    if spectra_num < 0:
        print('Invalid input number. Must be >= 0')
        return False

    # check for the data_keys
    allowed_keys = {"raw", "d1", "d2", "d3"}
    if not set(data_keys).issubset(allowed_keys) or len(data_keys) == 0:
        print(f'Invalid data_keys. Items must be from: {allowed_keys} and cannot be empty.')
        return False

    if shema != "BW" and len(data_keys) < 3:
        return False

    # Convert to numpy array
    spec_select = []
    for key in data_keys:
        spec_select.append([data[key].iloc[spectra_num].values])

    # Convert to respective Sheme
    if shema == 'BW':
        spec = encode_8bit(spec_select[0])
        array_bw = np.array([spec], dtype=np.uint8)
        img = Image.fromarray(array_bw, 'L')

    if shema == 'RGB':
        r_data = encode_8bit(spec_select[0])
        g_data = encode_8bit(spec_select[1])
        b_data = encode_8bit(spec_select[2])
        rgb = list(zip(r_data, g_data, b_data))
        rgb_array = np.array([rgb], dtype=np.uint8)
        img = Image.fromarray(rgb_array, 'RGB')

    if shema == 'HSV':
        h_data = encode_8bit(spec_select[0])
        s_data = encode_8bit(spec_select[1])
        v_data = encode_8bit(spec_select[2])
        hsv = list(zip(h_data, s_data, v_data))
        hsv_array = np.array([hsv], dtype=np.uint8)
        img = Image.fromarray(hsv_array, 'HSV')

    if shema == 'LAB':
        l_data = encode_8bit(spec_select[0])
        a_data = encode_8bit(spec_select[1])
        b_data = encode_8bit(spec_select[2])
        lab = list(zip(l_data, a_data, b_data))
        lab_array = np.array([lab], dtype=np.uint8)
        img = Image.fromarray(lab_array, 'LAB')

    #TODO: ATTENTION. EACH ROW IS SCALED! NOT SCALED TOGETHER=BETTER?
    if shema == 'L':
        l1 = encode_8bit(spec_select[0])
        l2 = encode_8bit(spec_select[1])
        l3 = encode_8bit(spec_select[2])
        val_array = np.array([l1,l2,l3], dtype=np.uint8)
        img = Image.fromarray(val_array, 'L')

    if shema == 'CMYK':
        c_data = encode_8bit(spec_select[0])
        m_data = encode_8bit(spec_select[1])
        y_data = encode_8bit(spec_select[2])
        k_data = encode_8bit(spec_select[3])
        lab = list(zip(c_data, m_data, y_data, k_data))
        lab_array = np.array([lab], dtype=np.uint8)
        img = Image.fromarray(lab_array, 'CMYK')

    return img

# See Pillow Modes (ranging from 0-255 for all 8bit modes)
def encode_8bit(data):
    scaler = MinMaxScaler((0,255))
    data = np.array(data).reshape(-1, 1)
    return scaler.fit_transform(data).flatten()
