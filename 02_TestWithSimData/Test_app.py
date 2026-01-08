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
def load_data():
    df = data_gen(n = 3, length=50, resolution=0.5, noise=1, num_peaks=1, seed=42)
    return df

data = load_data()

st.title("S2I-CNN - Test :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for 'simulated' data.")

tab_data, tab_results = st.tabs(["Data" ,"Results"])

with tab_data:
    st.header("Data")
    st.write("The first 10 rows of the generated data:")
    st.write("3 first spectra:")
    st.dataframe(data.head(3))

    fig, ax = plt.subplots()
    sns.lineplot(data=data.head(3).T, legend=False)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    st.pyplot(fig)


with tab_results:
    st.header("Results")

st.divider()

st.write("Lucas F. Voges, 2025")