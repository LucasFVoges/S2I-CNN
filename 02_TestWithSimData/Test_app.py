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
            max-width:50rem;
        }
    </style>
    """
)

@st.cache_data
def load_data(n = 10, length=50, resolution=0.5, noise=1, y_shift=0.02, num_peaks=1, num_peaks_addition=1, peak_randomised_amount= 0.1, normalize=True, normalize_individually=False, seed=67):
    df = data_gen(n, length, resolution, noise, y_shift, num_peaks, num_peaks_addition, peak_randomised_amount, normalize, normalize_individually, seed)
    return df

data = load_data()

st.title("S2I-CNN - Simulation :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for 'simulated' data.")

tab_about, tab_data, tab_S2I, tab_CNN = st.tabs(["About", "Data" ,"S2I", "CNN"], default="Data")

with tab_about:
    st.header("About S2I - CNN")
    st.write("Spectra2Image CNN is a noval approach that is designed to use minimal differences in spectral data to classify them accordingly.")
    st.write("This approach use image conversion from classiscal spectra tabular data with the goal to utilise pre-trained image classification Networks.")

    st.subheader("Future Improvements:")
    st.write("1. Add more classes to the generated spectra, so that it is not a two class problem.")

with tab_data:
    st.header("Data")

    st.write("The Data can be generated here. Some combinations may be corrected in the code eg. the number of peaks cant be greater than half the length of the spectrum.")

    data_gen_form = st.form("Data Generator")
    with data_gen_form:
        data_col1, data_col2 = st.columns(2)
        with data_col1:
            num = st.number_input("Number of Spectra per Class:", value=10, step=1, min_value=10, max_value=10000)
            len = st.number_input("Length of Spectra:", value=50, step=1, min_value=10, max_value=3000)
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
        data = load_data(num, len, res, noi, y_shifted, n_peaks, n_peaks_add, peak_rand, norm, norm_indiv, seeded)

    fig, ax = plt.subplots()
    sns.lineplot(data=data.head(10).T, legend=False, dashes=False)
    ax.set_ylim(ymin=0.0)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.set_title("Random Spectra Simulation")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.lineplot(data=data.tail(10).T, legend=False, dashes=False)
    ax.set_ylim(ymin=0.0)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.set_title("With Added Peak")
    st.pyplot(fig)

    st.write("The first 5 and last 5 rows of the generated data:")
    st.dataframe(data.head(5))
    st.dataframe(data.tail(5))

    # TODO: Make directly here the first and second derivative?

with tab_S2I:
    st.header("S2I")
    st.write("The generated data can be converted to images. They need to be saved before used in the CNN.")

    st.write("EXAMPLE ONLY: Just the first and last spectra displayed")

    img = image_encoder(data.iloc[0].values, "BW")
    if img:
        st.image(img, caption="First Spectra", use_container_width=True)
    else:
        st.error("Failed to generate image.")

    img = image_encoder(data.iloc[19].values, "BW")
    if img:
        st.image(img, caption="Last Spectra", use_container_width=True)
    else:
        st.error("Failed to generate image.")

with tab_CNN:
    st.header("CNN")

st.divider()

st.write("Lucas F. Voges, 2025")