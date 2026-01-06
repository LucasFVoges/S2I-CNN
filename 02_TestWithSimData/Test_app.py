# Imports:
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from functions.ImageEncoder import image_encoder
from functions.DataGenerator import data_gen

@st.cache_data
def load_data():
    df = data_gen(type=1, length=10, seed=42)
    return df

data = load_data()

st.title("S2I-CNN - Test :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for example data.")

tab_data, tab_results = st.tabs(["Data", "Results"])

with tab_data:
    st.header("Data")
    st.write("text text tex")
    st.dataframe(data.head(10))

with tab_results:
    st.header("Results")

st.divider()

st.write("Lucas F. Voges, 2025")