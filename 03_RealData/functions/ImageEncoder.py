import os.path
import numpy as np
import colorsys
import math
from PIL import Image, ImageDraw
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
    :return: returns a PIL image.
    :rtype: image
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


def spider_image_encoder(data, data_keys = ["raw", "d1", "d2", "d3"], shema='HSV', spectra_num=0, size=224, margin=8, draw_outline = True, outline_width= 1):
    """
    Encodes one spectrum as a filled "spider/radar" plot image (directly size x size, default 224x224).

    :param data: Dictionary-like object where each key maps to a pandas DataFrame row-accessible via `.iloc[spectra_num].values`.
    :type data: dict
    :param spectra_num: Which spectrum (row) to encode.
    :type spectra_num: int
    :param data_keys: List of keys in the dictionary.
    :type data_keys: list of str
    :param shema: Color Shema 'BW', 'RGB', 'CMYK', 'HSV', 'LAB', 'L' (grayscale 2D) - (default: 'HSV').
    :type shema: str
    :param size: Output image size in pixels (square).
    :type size: int
    :param margin: Padding to keep the plot away from the border.
    :type margin : int
    :param draw_outline: Draw a shaped outline around the plot (optional).
    :type draw_outline: bool
    :param outline_width: how thick the outline should be (optional).
    :type outline_width: int

    Returns PIL.Image.Image (mode "RGB")
    """

    def _as_1d_float_array(key: str) -> np.ndarray:
        arr = np.asarray(data[key].iloc[spectra_num].values, dtype=np.float32).reshape(-1)
        return arr

    def _normalize_unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or abs(x_max - x_min) < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        y = (x - x_min) / (x_max - x_min)
        y = np.clip(y, 0.0, 1.0)
        return y.astype(np.float32)

    # checking input
    shema = shema.upper()
    if shema not in ['BW','RGB', 'CMYK', 'HSV', 'LAB', 'L']:
        print('Invalid color shema.')
        return False

    if shema == 'BW' or shema == 'HSV':
        if len(data_keys) < 2:
            print("At least 1 additional data keys must be provided.")
            return False
    else:
        if len(data_keys) < 4:
            print("At least 4 additional data keys must be provided.")
            return False

    if spectra_num < 0:
        raise ValueError("spectra_num must be >= 0")

    for k in data_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in data.")

    # Convert to numpy array
    spec_select = []
    for key in data_keys:
        spec_select.append(_as_1d_float_array(key))

    n = len(spec_select[0])
    if n < 3:
        raise ValueError(f"Spectrum length must be >= 3 for a spider plot, got {n}")

    len_spec = len(spec_select)
    # Normalize to [0..1]
    r = _normalize_unit(spec_select[0])
    r1 = _normalize_unit(spec_select[1])
    if len_spec > 2:
        r2 = _normalize_unit(spec_select[2])
    if len_spec > 3:
        r3 = _normalize_unit(spec_select[3])

    img = Image.new("RGB", (size, size), (255,105,180))
    draw = ImageDraw.Draw(img)

    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    max_radius = (min(size, size) / 2.0) - margin

    # Use a clean, deterministic ordering around the circle.
    # Start at -90° (top) and go clockwise.
    start_angle = -math.pi / 2.0

    # Precompute points
    points = []
    for i in range(n):
        ang = start_angle + (2.0 * math.pi) * (i / n)
        rr = float(r[i]) * max_radius
        x = cx + rr * math.cos(ang)
        y = cy + rr * math.sin(ang)
        points.append((x, y))

    # Draw as colored wedges (triangles) between consecutive points and the center.
    for i in range(n):
        j = (i + 1) % n
        if shema == 'BW':
            g = int(round(255.0 * r1[i]))
            color = (g, g, g)
        elif shema == "RGB":
            # RGB per sector from variables (same pattern as HSV, but direct channels)
            if len_spec > 3:
                color = (int(round(255.0 * r1[i])),
                         int(round(255.0 * r2[i])),
                         int(round(255.0 * r3[i])))
            else:
                return False
        elif shema == 'HSV':
            # HSV -> RGB per sector (using derivative-derived values at i)
            if len_spec > 3:
                rgb_float = colorsys.hsv_to_rgb(r1[i], r2[i], r3[i])
            elif len_spec > 2:
                rgb_float = colorsys.hsv_to_rgb(r1[i], r2[i], 1)
            else:
                rgb_float = colorsys.hsv_to_rgb(r1[i], 1, 1)
            color = tuple(round(255.0 * c) for c in rgb_float)
        elif shema == 'LAB':
                return False
        else:
            return False

        tri = [(cx, cy), points[i], points[j]]
        draw.polygon(tri, fill=color)

    if draw_outline:
        # Outline the outer polygon for extra shape definition (optional)
        draw.line(points + [points[0]], fill=(0,0,0), width=int(outline_width))

    return img

# See Pillow Modes (ranging from 0-255 for all 8bit modes)
def encode_8bit(data):
    scaler = MinMaxScaler((0,255))
    data = np.array(data).reshape(-1, 1)
    return scaler.fit_transform(data).flatten()
