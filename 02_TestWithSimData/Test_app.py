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

@st.cache_data
def load_data(n = 10, length=50, resolution=0.5, noise=1, y_shift=0.02, num_peaks=3, peak_randomised_amount= 0.1, seed=12):
    df = data_gen(n, length, resolution, noise, y_shift, num_peaks, peak_randomised_amount, seed)
    return df

data = load_data()

st.title("S2I-CNN - Simulation :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for 'simulated' data.")

tab_data, tab_S2I, tab_CNN = st.tabs(["Data" ,"S2I", "CNN"])

with tab_data:
    st.header("Data")

    data_gen_form = st.form("Data Generator")
    with data_gen_form:
        num = st.number_input("Number of Spectra per Class:", value=10, step=1, min_value=1, max_value=1000)
        len = st.number_input("Length of Spectra:", value=50, step=1, min_value=10, max_value=1000)
        n_peaks = st.number_input("Number of peaks:", value=3, step=1, min_value=1, max_value=100)
        noi = st.slider("Noise:", 0.0, 10.0, 1.0)
        y_shifted = st.slider("Shift:", 0.00, 1.00, 0.02)
        peak_rand = st.slider("Randomised:", 0.0, 1.0, 0.1)
        rand = st.checkbox("Random Seed")
        submit = st.form_submit_button("generate new data")

    if submit:
        if rand:
            seeded = np.random.randint(1,1000)
        else:
            seeded = 12
        data = load_data(num, len, 0.5, noi, y_shifted, n_peaks, peak_rand, seeded)

    fig, ax = plt.subplots()
    sns.lineplot(data=data.head(10).T, legend=False, dashes=False)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.set_title("Random Spectra Simulation")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.lineplot(data=data.tail(10).T, legend=False, dashes=False)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.set_title("With Added Peak")
    st.pyplot(fig)

    st.write("The first 5 and last 5 rows of the generated data:")
    st.dataframe(data.head(5))
    st.dataframe(data.tail(5))

with tab_S2I:
    st.header("S2I")

with tab_CNN:
    st.header("CNN")

st.divider()

st.write("Lucas F. Voges, 2025")