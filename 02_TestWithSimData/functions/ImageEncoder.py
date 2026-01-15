import os.path
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    pass

def image_encoder(data, shema='RGB'):
    """
    Converts spectral intensities to an Image. (1 Spectra)

    :param data: Spectral intensities.
    :type data: array of num
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

    # Convert to numpy array if it's a list or pandas object
    data = np.asarray(data)

    # if len(data) < 3:
    #     print('Invalid data length. len is ', len(data), 'but should be 3')
    #     return False

    # Convert to respective Sheme
    if shema == 'BW':
        data = encode_8bit(data)
        array_bw = np.array([data], dtype=np.uint8)
        img = Image.fromarray(array_bw, 'L')

    if shema == 'RGB':
        r_data = encode_8bit(data[0])
        g_data = encode_8bit(data[1])
        b_data = encode_8bit(data[2])
        rgb = list(zip(r_data, g_data, b_data))
        rgb_array = np.array([rgb], dtype=np.uint8)
        img = Image.fromarray(rgb_array, 'RGB')

    if shema.upper() == 'HSV':
        h_data = encode_8bit(data[0])
        s_data = encode_8bit(data[1])
        v_data = encode_8bit(data[2])
        hsv = list(zip(h_data, s_data, v_data))
        hsv_array = np.array([hsv], dtype=np.uint8)
        img = Image.fromarray(hsv_array, 'HSV')

    if shema.upper() == 'LAB':
        l_data = encode_8bit(data[0])
        a_data = encode_8bit(data[1])
        b_data = encode_8bit(data[2])
        lab = list(zip(l_data, a_data, b_data))
        lab_array = np.array([lab], dtype=np.uint8)
        img = Image.fromarray(lab_array, 'LAB')

    if shema.upper() == 'CMYK':
        if len(data) < 4:
            print('Invalid data length. len is ', len(data), 'but should be 4')
            return False
        c_data = encode_8bit(data[0])
        m_data = encode_8bit(data[1])
        y_data = encode_8bit(data[2])
        k_data = encode_8bit(data[3])
        lab = list(zip(c_data, m_data, y_data, k_data))
        lab_array = np.array([lab], dtype=np.uint8)
        img = Image.fromarray(lab_array, 'CMYK')

    #TODO: ATTENTION. EACH ROW IS SCALED! NOT SCALED TOGETHER=BETTER?
    if shema.upper() == 'L':
        l1 = encode_8bit(data[0])
        l2 = encode_8bit(data[1])
        l3 = encode_8bit(data[2])
        val_array = np.array([l1,l2,l3], dtype=np.uint8)
        img = Image.fromarray(val_array, 'L')

    return img

# See Pillow Modes (ranging from 0-255 for all 8bit modes)
def encode_8bit(data):
    scaler = MinMaxScaler((0,255))
    data = np.array(data).reshape(-1, 1)
    return scaler.fit_transform(data).flatten()
