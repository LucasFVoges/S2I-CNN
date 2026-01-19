# Imports:
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from functions.ImageEncoder import image_encoder
from functions.DataGenerator import data_gen

# Set the dark style
plt.style.use("dark_background")
# Set layout
st.set_page_config(layout="centered")
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:56rem;
        }
    </style>
    """
)

@st.cache_data
def load_data(n = 10, length=50, resolution=0.5, noise=1, y_shift=0.02, num_peaks=1, num_peaks_addition=1, peak_randomised_amount= 0.1, normalize=True, normalize_individually=False, seed=67):
    df = data_gen(n, length, resolution, noise, y_shift, num_peaks, num_peaks_addition, peak_randomised_amount, normalize, normalize_individually, seed)

    # Calculate derivatives
    d1 = pd.DataFrame(np.gradient(df.values, axis=1), columns=df.columns, index=df.index)
    d2 = pd.DataFrame(np.gradient(d1.values, axis=1), columns=df.columns, index=df.index)
    d3 = pd.DataFrame(np.gradient(d2.values, axis=1), columns=df.columns, index=df.index)

    return {"raw": df, "d1": d1, "d2": d2, "d3": d3}

data = load_data()

st.title("S2I-CNN - Simulation :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for 'simulated' data.")

tab_about, tab_data, tab_S2I, tab_cluster, tab_CNN = st.tabs(["About", "Data" ,"S2I", "Clustering", "CNN"], default="Data")

with tab_about:
    st.header("About S2I - CNN")
    st.write("Spectra2Image CNN is a noval approach that is designed to use minimal differences in spectral data to classify them accordingly.")
    st.write("This approach use image conversion from classical spectra tabular data with the goal to utilise pre-trained image classification Networks.")

    st.subheader("Future Improvements:")
    st.write("1. Add more classes to the generated spectra, so that it is not a two class problem.")
    st.write("2. Change the derivative function to a more complex Savitzky-Golay function.")
    st.write("3. Add more images from the conversion (at the moment just first and last spectra are shown from the set.")
    st.write("4. Add a selection so that only a part (fingerprint area) of the spectra will be used...")
    st.write("5. Add more image conversion models.")

with tab_data:
    st.header("Data")

    st.write("The Data can be generated here. Some combinations may be corrected in the code eg. the number of peaks cant be greater than half the length of the spectrum.")

    data_gen_form = st.form("Data Generator")
    with data_gen_form:
        data_col1, data_col2 = st.columns(2)
        with data_col1:
            num = st.number_input("Number of Spectra per Class:", value=10, step=1, min_value=10, max_value=10000)
            leng = st.number_input("Length of Spectra:", value=50, step=1, min_value=10, max_value=3000)
            noi = st.slider("Noise:", 0.01, 10.00, 1.0)
            y_shifted = st.slider("Shift:", 0.00, 1.00, 0.02)
            rand = st.checkbox("Random Seed? (else: 67)")
        with data_col2:
            n_peaks = st.number_input("Number of peaks:", value=1, step=1, min_value=1, max_value=1000)
            n_peaks_add = st.number_input("Number of Added peaks (Second Class):", value=1, step=1, min_value=1, max_value=100)
            peak_rand = st.slider("Randomised:", 0.0, 1.0, 0.1)
            res = st.slider("Resolution:", 0.1, 1.0, 0.5, step=0.1, format="%0.1f")
            norm = st.checkbox("Normalize Spectra [0,1]", True)
            norm_indiv = st.checkbox("When normalized: every Spectra individually", False)

        submit = st.form_submit_button("generate new data")

    if submit:
        if rand:
            seeded = np.random.randint(1,1000)
        else:
            seeded = 67
        data = load_data(num, leng, res, noi, y_shifted, n_peaks, n_peaks_add, peak_rand, norm, norm_indiv, seeded)

    spec_tab, deriv_tab, deriv_2_tab, deriv_3_tab = st.tabs(["Spectra", "Spectra 1st Derivative", "Spectra 2nd Derivative", "Spectra 3rd Derivative"])

    def show_spectral_data(dtype = "raw", ymin0=True):
        number_of_plots = num if num < 100 else 100

        col_spec_img1, col_spec_img2 = st.columns(2)
        with col_spec_img1:
            fig, ax = plt.subplots()
            sns.lineplot(data=data[dtype].head(number_of_plots).T, legend=False, dashes=False)
            if ymin0: ax.set_ylim(ymin=0.0)
            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Intensity")
            ax.set_title("Random Spectra Simulation")
            st.pyplot(fig)

        with col_spec_img2:
            fig, ax = plt.subplots()
            sns.lineplot(data=data[dtype].tail(number_of_plots).T, legend=False, dashes=False)
            if ymin0: ax.set_ylim(ymin=0.0)
            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Intensity")
            ax.set_title("With Added Peak")
            st.pyplot(fig)

        st.write("The first 5 and last 5 rows of the generated data:")
        st.dataframe(data[dtype].head(5))
        st.dataframe(data[dtype].tail(5))

    with spec_tab:
        st.write("**WARNING:** max 100 spectra are shown!")
        show_spectral_data("raw", norm)

    with deriv_tab:
        show_spectral_data("d1", False)

    with deriv_2_tab:
        show_spectral_data("d2", False)

    with deriv_3_tab:
        show_spectral_data("d3", False)

with tab_S2I:
    st.header("S2I")
    st.write("**WARNING:** Images displayed here are not the original images! "
             "Due to anti-aliasing and the display in a visible size in the browser, "
             "the Images may seem blurred or in a different size."
             "**Selection** of the data (derivatives) is important, as it is used in the order selected. For BW (Black and White) model, only the first selection is used..." )
    st.write("*ll conversion models use 3 input data! Except BW which uses just one channel.")

    # Local variables
    x_size, y_size = 10000, 1000

    # Input:
    data_which_deriv = st.multiselect("Which spectra data should be used? (order matters!)", ["0st Derivative", "1st Derivative", "2st Derivative", "3st Derivative"], default=["0st Derivative", "1st Derivative", "2st Derivative"])
    conv = st.selectbox("Select the conversion model:", ('BW: Black and White','RGB: Red Green Blue', 'HSL: Hue Saturation Lightness', 'LAB: Lightness A B'))
    # todo: check the input dynamically, so that only models can be selected that are compatible with the data!

    # Mapping:
    mapping_data = {"0st Derivative": "raw", "1st Derivative": "d1", "2st Derivative": "d2", "3st Derivative": "d3"}
    selection = []
    for i in data_which_deriv: selection.append(mapping_data[i])

    # Mapping conv:
    mapping_conv = {"BW: Black and White": "BW", "RGB: Red Green Blue": "RGB", "HSL: Hue Saturation Lightness": "HSV", "LAB: Lightness A B": "LAB"}

    st.subheader(f"Image of Spectra for {conv} Conversion:")

    # Image for 1st class
    img = image_encoder(data, selection, 0, mapping_conv[conv])
    if img:
        st.image(img.convert('RGB').resize((x_size, y_size), Image.Resampling.BOX), caption="First Spectra", width="stretch")
    else:
        st.error("Failed to generate image.")

    # Image for 2nd class
    img = image_encoder(data, selection, data["raw"].shape[0] - 1, mapping_conv[conv])
    if img:
        st.image(img.convert('RGB').resize((x_size, y_size), Image.Resampling.BOX), caption="Last Spectra", width="stretch")
    else:
        st.error("Failed to generate image.")


with tab_cluster:
    st.header("Clustering")
    st.write("Classic clustering machine learning algorithms to have a baseline for the classification performance.")

with tab_CNN:
    st.header("CNN")

st.divider()

st.write("Lucas F. Voges, 2026")