# Imports:
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from functions.ImageEncoder import image_encoder
from functions.DataGenerator import data_gen
from functions.Clustering import cluster_pca, cluster_tsne, cluster_kmeans, cluster_dbscan, classify_svm

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
def load_data(n = 10, length=50, resolution=0.5, noise=1, y_shift=0.02, num_peaks=1, num_peaks_addition=1, peak_randomised_amount= 0.1, normalize=True, normalize_individually="Vector", seed=67):
    df = data_gen(n, length+12, resolution, noise, y_shift, num_peaks, num_peaks_addition, peak_randomised_amount, normalize, normalize_individually, seed)

    # Calculate derivatives with Savitzky-Golay:
    window = 12
    poly = 2
    trim = 6

    d0_vals = savgol_filter(df.values, window_length=window, polyorder=poly, deriv=0, axis=1)
    d1_vals = savgol_filter(df.values, window_length=window, polyorder=poly, deriv=1, axis=1)
    d2_vals = savgol_filter(df.values, window_length=window, polyorder=poly+1, deriv=2, axis=1)
    d3_vals = savgol_filter(df.values, window_length=window, polyorder=poly+2, deriv=3, axis=1)

    d0 = pd.DataFrame(d0_vals, columns=df.columns, index=df.index).iloc[:, trim:-trim]
    d1 = pd.DataFrame(d1_vals, columns=df.columns, index=df.index).iloc[:, trim:-trim]
    d2 = pd.DataFrame(d2_vals, columns=df.columns, index=df.index).iloc[:, trim:-trim]
    d3 = pd.DataFrame(d3_vals, columns=df.columns, index=df.index).iloc[:, trim:-trim]

    # Alternative implementation of the derivative calculation:
    # d1 = pd.DataFrame(np.gradient(df.values, axis=1), columns=df.columns, index=df.index)
    # d2 = pd.DataFrame(np.gradient(d1.values, axis=1), columns=df.columns, index=df.index)
    # d3 = pd.DataFrame(np.gradient(d2.values, axis=1), columns=df.columns, index=df.index)

    return {"raw": d0, "d1": d1, "d2": d2, "d3": d3}

if 'data' not in st.session_state:
    st.session_state.data = load_data()
data = st.session_state.data

st.title("S2I-CNN - Simulation :streamlit:")
st.write("An example of how the 'Spectra2Image CNN' can be used for 'simulated' data.")

tab_about, tab_data, tab_cluster, tab_S2I, tab_CNN = st.tabs(["About", "Data", "Clustering" ,"S2I", "CNN"], default="Data")

with tab_about:
    st.header("About S2I - CNN")
    st.write("Spectra2Image CNN is a noval approach that is designed to use minimal differences in spectral data to classify them accordingly.")
    st.write("This approach use image conversion from classical spectra tabular data with the goal to utilise pre-trained image classification Networks.")

    st.subheader("Future Improvements:")
    st.write("1. Add more classes to the generated spectra, so that it is not a two class problem.")
    st.write("4. Add more image conversion models.")
    st.write("5. Generator can include baseline drift instead of simple y-shift.")
    st.write("6. Add peaks to both classes instead of just one.")
    st.write("7. Add clustering for 1st derivative data for all methods!")

    st.write("For high-dimensional sparse data it is helpful to first reduce the dimensions to 50 dimensions with `TruncatedSVD` and then perform t-SNE. This will usually improve the visualization.")

with tab_data:
    st.header("Data")

    norm_help = """
    **Normalization Methods:**
    *applied to each spectrum individually*
    - **Vector (L2):** $y_i = \\frac{x_i}{\\sqrt{\\sum x_i^2}}$
    - **SNV (Standard Normal Variate):** $y_i = \\frac{x_i - \\bar{x}}{s}$
    - **MinMax:** $y_i = \\frac{x_i - min(x)}{max(x) - min(x)}$
    """

    st.write("The Data can be generated here. Some combinations may be corrected. A Savitzky-Golay filter is applied to the spectra.")

    data_gen_form = st.form("Data Generator")
    with data_gen_form:
        data_col1, data_col2 = st.columns(2)
        with data_col1:
            num = st.number_input("Number of Spectra per Class:", value=10, step=1, min_value=10, max_value=10000)
            leng = st.number_input("Length of Spectra:", value=50, step=1, min_value=20, max_value=3000)
            noi = st.slider("Noise:", 0.01, 10.00, 1.0)
            y_shifted = st.slider("Shift:", 0.00, 1.00, 0.02)
            rand = st.checkbox("Random Seed? (else: 67)")
        with data_col2:
            n_peaks = st.number_input("Number of peaks:", value=1, step=1, min_value=1, max_value=1000)
            n_peaks_add = st.number_input("Number of Added peaks (Second Class):", value=1, step=1, min_value=1, max_value=100)
            peak_rand = st.slider("Randomised:", 0.0, 1.0, 0.1, help="Strength of peak width, position and amplitude randomisation.")
            res = st.slider("Resolution:", 0.1, 1.0, 0.5, step=0.1, format="%0.1f", help="length / resolution = datapoints")
            norm = st.checkbox("MinMax Scaling", True, help="Scale the whole dataset between 0 and 1.")
            norm_indiv = st.selectbox("Normalisation", ("None", "Vector", "SNV", "MinMax"), index=1,help=norm_help)

        submit = st.form_submit_button("generate new data")

    if submit:
        if rand:
            seeded = np.random.randint(1,1000)
        else:
            seeded = 67
        st.session_state.data = load_data(num, leng, res, noi, y_shifted, n_peaks, n_peaks_add, peak_rand, norm, norm_indiv, seeded)
        st.session_state.pca_df = None
        st.session_state.tsne_df = None
        st.session_state.pca_variance = None
        st.session_state.kmeans_results = None
        st.session_state.dbscan_results = None
        st.session_state.svm_results = None
        st.rerun()

    spec_tab, deriv_tab, deriv_2_tab, deriv_3_tab = st.tabs(["Spectra", "Spectra 1st Derivative", "Spectra 2nd Derivative", "Spectra 3rd Derivative"])

    def show_spectral_data(dtype = "raw", ymin0=True):
        number_of_plots = num if num <= 99 else 99

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




with tab_cluster:
    st.header("Clustering")
    st.write("Classic clustering and classification machine learning algorithms to have a baseline for the classification performance.")

    # Initialize session state for clustering results
    if 'pca_df' not in st.session_state:
        st.session_state.pca_df = None
        st.session_state.pca_variance = None
    if 'tsne_df' not in st.session_state:
        st.session_state.tsne_df = None
    if 'kmeans_results' not in st.session_state:
        st.session_state.kmeans_results = None
    if 'dbscan_results' not in st.session_state:
        st.session_state.dbscan_results = None
    if 'svm_results' not in st.session_state:
        st.session_state.svm_results = None

    mapping_data = {"Spectra": "raw", "1st Derivative": "d1"}

    st.subheader("PCA:")
    st.write("A simple PCA scores plot. When classes can be directly seen, the generated data may be to simple...")

    if st.button("Run PCA"):
        pca_data = data["raw"]
        n_samples = len(pca_data)
        labels = ["First Class"] * (n_samples // 2) + ["Second Class"] * (n_samples // 2)

        st.session_state.pca_df, st.session_state.pca_variance = cluster_pca(pca_data, labels)

    if st.session_state.pca_df is not None:
        fig_pca, ax_pca = plt.subplots()
        sns.scatterplot(data=st.session_state.pca_df, x="PC1", y="PC2", hue="Class", palette="viridis", legend=False,
                        ax=ax_pca)
        var = st.session_state.pca_variance
        ax_pca.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax_pca.set_ylabel(f"PC2 ({var[1]:.1%})")
        st.pyplot(fig_pca)
        st.write(f"Total explained variance by first two components: {sum(var):.1%}")


    st.subheader("T-SNE Clustering:")
    if st.button("Run T-SNE Clustering"):
        tsne_data = data["raw"]
        n_samples = len(tsne_data)
        labels = ["First Class"] * (n_samples // 2) + ["Second Class"] * (n_samples - n_samples // 2)

        st.session_state.tsne_df = cluster_tsne(tsne_data, labels)

    if st.session_state.tsne_df is not None:
        fig_tsne, ax_tsne = plt.subplots()
        sns.scatterplot(data=st.session_state.tsne_df, x="TSNE1", y="TSNE2", hue="Class", palette="viridis", legend=False, ax=ax_tsne)
        st.pyplot(fig_tsne)



    st.subheader("K-Means Clustering:")
    if st.button("Run K-Means Clustering"):
        km_data = data["raw"]
        n_samples = len(km_data)
        labels = ["First Class"] * (n_samples // 2) + ["Second Class"] * (n_samples // 2)

        st.session_state.kmeans_results = cluster_kmeans(km_data, labels, n_clusters=2)

    if st.session_state.kmeans_results is not None:
        actual_means, predicted_means, confusion_matrix, metrics = st.session_state.kmeans_results

        st.write("Comparing the average spectra of actual classes vs. K-Means identified clusters:")
        col1, col2 = st.columns(2)

        with col1:
            fig_act, ax_act = plt.subplots()
            sns.lineplot(data=actual_means.T, legend=False, ax=ax_act)
            ax_act.set_title("Actual Class Averages")
            ax_act.set_ylabel("Intensity")
            st.pyplot(fig_act)

        with col2:
            fig_pred, ax_pred = plt.subplots()
            sns.lineplot(data=predicted_means.T, legend=False, ax=ax_pred)
            ax_pred.set_title("K-Means Cluster Centroids")
            ax_pred.set_ylabel("Intensity")
            st.pyplot(fig_pred)

        st.write("Clustering Performance Metrics:")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        m_col2.metric("Precision", f"{metrics['Precision']:.1%}")
        m_col3.metric("Recall", f"{metrics['Recall']:.1%}")
        m_col4.metric("F1-Score", f"{metrics['F1-Score']:.1%}")

        st.write("**Confusion Matrix:**")
        st.table(confusion_matrix)

        st.info("Note: Clusters can be switched as this is not supervised learning! They are selected as correct by majority vote.")

    st.subheader("DBSCAN Clustering:")
    st.info(":warning: NOT PROPERLY WORKING :warning:")
    with st.form("dbscan_form"):
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            eps = st.number_input("Epsilon (eps):", value=0.5, step=0.1, min_value=0.01)
        with db_col2:
            min_s = st.number_input("Min Samples:", value=5, step=1, min_value=1)
        db_submit = st.form_submit_button("Run DBSCAN")

    if db_submit:
        db_data = data["raw"]
        n_samples = len(db_data)
        labels = ["First Class"] * (n_samples // 2) + ["Second Class"] * (n_samples // 2)
        st.session_state.dbscan_results = cluster_dbscan(db_data, labels, eps=eps, min_samples=min_s)

    if st.session_state.dbscan_results is not None:
        df_db, db_metrics = st.session_state.dbscan_results

        st.write("Clustering Performance Metrics (Mapped to Classes):")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{db_metrics['Accuracy']:.1%}")
        m_col2.metric("Precision", f"{db_metrics['Precision']:.1%}")
        m_col3.metric("Recall", f"{db_metrics['Recall']:.1%}")
        m_col4.metric("F1-Score", f"{db_metrics['F1-Score']:.1%}")

        st.write(f"DBSCAN Specifics: Found {db_metrics['Clusters Found']} clusters and identified {db_metrics['Noise Points']} noise points.")

        st.write("Average Spectra per DBSCAN Cluster:")

        # Calculate means for the clusters found (excluding raw labels)
        cluster_means = df_db.groupby("Cluster").mean(numeric_only=True).drop(columns=[], errors='ignore')

        fig_db_spec, ax_db_spec = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=cluster_means.T, dashes=False, ax=ax_db_spec)
        ax_db_spec.set_title("DBSCAN Cluster Averages")
        ax_db_spec.set_ylabel("Intensity")
        ax_db_spec.set_xlabel("Wavenumber")
        st.pyplot(fig_db_spec)

        if db_metrics["Noise Points"] > 0:
            st.info(
                f"Note: The 'Noise' line represents the average of the {db_metrics['Noise Points']} points that didn't fit into any cluster.")

    st.subheader("SVM Classification Baseline:")
    st.write("Supervised baseline (70/30 Train/Test split).")

    with st.form("svm_form"):
        svm_col1, svm_col2 = st.columns(2)
        with svm_col1:
            kernel = st.selectbox("Kernel:", ("linear", "poly", "rbf", "sigmoid"), help="Determines the shape of the decision boundary. 'linear' is simple, 'rbf' handles complex non-linear patterns.")
        with svm_col2:
            c_val = st.number_input("C (Regularization):", value=1.0, min_value=0.01, step=0.1, help="Controls the trade-off between smooth boundary and classifying training points correctly. Smaller C = smoother boundary (less overfitting).")
            data_channel = st.selectbox("Select the data:", ("Spectra", "1st Derivative"), index=0)
        svm_submit = st.form_submit_button("Run SVM Classification")

    if svm_submit:
        svm_data = data[mapping_data[data_channel]]
        n_samples = len(svm_data)
        labels = ["First Class"] * (n_samples // 2) + ["Second Class"] * (n_samples // 2)
        st.session_state.svm_results = classify_svm(svm_data, labels, kernel=kernel, c=c_val)

    if st.session_state.svm_results is not None:
        svm_train, svm_test, cm_train, cm_test = st.session_state.svm_results

        st.write("**Training Set Metrics:**")
        tr_col1, tr_col2, tr_col3, tr_col4 = st.columns(4)
        tr_col1.metric("Accuracy", f"{svm_train['Accuracy']:.1%}")
        tr_col2.metric("Precision", f"{svm_train['Precision']:.1%}")
        tr_col3.metric("Recall", f"{svm_train['Recall']:.1%}")
        tr_col4.metric("F1-Score", f"{svm_train['F1-Score']:.1%}")

        st.write("**Test Set Metrics:**")
        te_col1, te_col2, te_col3, te_col4 = st.columns(4)
        te_col1.metric("Accuracy", f"{svm_test['Accuracy']:.1%}")
        te_col2.metric("Precision", f"{svm_test['Precision']:.1%}")
        te_col3.metric("Recall", f"{svm_test['Recall']:.1%}")
        te_col4.metric("F1-Score", f"{svm_test['F1-Score']:.1%}")

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.write("**Confusion Matrix (Train):**")
            st.table(cm_train)

        with col_cm2:
            st.write("**Confusion Matrix (Test):**")
            st.table(cm_test)


with tab_S2I:
    st.header("S2I")
    st.info("**WARNING:** Images displayed here are not the original images! "
             "Due to anti-aliasing and the display in a visible size in the browser, "
             "the Images may seem blurred or in a different size. "
             "**Selection** of the data (derivatives) is important, as it is used in the order selected. For BW (Black and White) model, only the first selection is used..." )

    # Local variables
    x_size, y_size = 1000, 40

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

    # Image examples
    st.text("First Class:")
    for i in range(10):
        img = image_encoder(data, selection, i, mapping_conv[conv])
        if img:
            converted_img = img.convert('RGB').resize((x_size, y_size), Image.Resampling.BOX)
            st.image(converted_img, width="stretch")
        else:
            st.error("Failed to generate image.")
            break
    st.text("Second Class:")
    for i in range(10):
        img = image_encoder(data, selection, data["raw"].shape[0] - (i+1), mapping_conv[conv])
        if img:
            converted_img = img.convert('RGB').resize((x_size, y_size), Image.Resampling.BOX)
            st.image(converted_img, width="stretch")
        else:
            st.error("Failed to generate image.")
            break

with tab_CNN:
    st.header("CNN")

st.divider()

st.write("Lucas F. Voges, 2026")