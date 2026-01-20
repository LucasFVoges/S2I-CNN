import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

if __name__ == "__main__":
    pass

def cluster_pca(data, labels):
    """
        "clusters" the spectral data using a pca

        :param data: spectral data in form of an array
        :type data: array
        :param labels: labels for the data
        :type labels: array

    """
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)

    df = pd.DataFrame(pca_results, columns=["PC1", "PC2"])
    df["Class"] = labels
    return df, pca.explained_variance_ratio_


def cluster_tsne(data, labels):
    """
    Clusters the spectral data using T-SNE
    """

    perplexity = 10 if len(data) < 50 else 30
    tsne = TSNE(n_components=2, perplexity=perplexity) #random_state=67
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
    df["Class"] = labels
    return df


def cluster_kmeans(data, labels, n_clusters=2):
    """
    Clusters the spectral data using K-Means and returns centroids/averages
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto') #random_state=67,
    cluster_labels = kmeans.fit_predict(data)

    mapping_df = pd.DataFrame({
        "Actual": labels,
        "Cluster": cluster_labels
    })
    # Map Cluster IDs to Actual Labels
    mapping = {}
    for i in range(n_clusters):
        cluster_subset = mapping_df[mapping_df["Cluster"] == i]
        if not cluster_subset.empty:
            majority_class = cluster_subset["Actual"].value_counts().idxmax()
            mapping[i] = majority_class
        else:
            mapping[i] = f"Empty Cluster {i}"

    predicted_labels = [mapping[c] for c in cluster_labels]


    # Metrics
    metrics = {
        "Accuracy": accuracy_score(labels, predicted_labels),
        "Precision": precision_score(labels, predicted_labels, pos_label="Second Class", zero_division=0),
        "Recall": recall_score(labels, predicted_labels, pos_label="Second Class", zero_division=0),
        "F1-Score": f1_score(labels, predicted_labels, pos_label="Second Class", zero_division=0)
    }

    # Calculate averages for plotting
    df_temp = data.copy()
    df_temp["Actual Class"] = labels
    actual_means = df_temp.groupby("Actual Class").mean(numeric_only=True)

    df_temp["Predicted Cluster"] = [f"Cluster {c} ({mapping[c]})" for c in cluster_labels]
    predicted_means = df_temp.groupby("Predicted Cluster").mean(numeric_only=True)

    # Confusion matrix
    confusion_matrix = pd.crosstab(df_temp["Actual Class"], df_temp["Predicted Cluster"])

    return actual_means, predicted_means, confusion_matrix, metrics


def cluster_dbscan(data, labels, eps=0.5, min_samples=5):
    """
    Clusters the spectral data using DBSCAN
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)

    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    mapping_df = pd.DataFrame({
        "Actual": labels,
        "Cluster": cluster_labels
    })

    # Map clusters to majority class
    mapping = {}
    for cluster_id in unique_clusters:
        cluster_subset = mapping_df[mapping_df["Cluster"] == cluster_id]
        majority_class = cluster_subset["Actual"].value_counts().idxmax()
        mapping[cluster_id] = majority_class

    # Handle noise: usually noise is counted as a 'wrong' prediction
    # for metric purposes, we'll map it to a specific label
    mapping[-1] = "Noise/Unclassified"

    predicted_labels = [mapping[c] for c in cluster_labels]

    metrics = {
        "Accuracy": accuracy_score(labels, predicted_labels),
        "Precision": precision_score(labels, predicted_labels, average='weighted', zero_division=0),
        "Recall": recall_score(labels, predicted_labels, average='weighted', zero_division=0),
        "F1-Score": f1_score(labels, predicted_labels, average='weighted', zero_division=0),
        "Noise Points": list(cluster_labels).count(-1),
        "Clusters Found": len(unique_clusters)
    }

    df_results = data.copy()
    df_results["Actual Class"] = labels
    df_results["Cluster"] = [f"Cluster {c} ({mapping[c]})" if c != -1 else "Noise" for c in cluster_labels]

    return df_results, metrics

def classify_svm(data, labels, kernel='linear', c=1.0):
    """
    Classifies the spectral data using a Support Vector Machine (SVM)
    """
    # Split data for a realistic "baseline" evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=67, stratify=labels
    )

    clf = SVC(kernel=kernel, C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Function to bundle metrics to avoid repetition
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label="Second Class", zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label="Second Class", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, pos_label="Second Class", zero_division=0)
        }

    metrics_train = get_metrics(y_train, y_pred_train)
    metrics_test = get_metrics(y_test, y_pred_test)

    # Confusion matrix for training set
    cm_train = pd.crosstab(
        pd.Series(y_train, name='Actual'),
        pd.Series(y_pred_train, name='Predicted')
    )

    # Confusion matrix for test set
    cm_test = pd.crosstab(
        pd.Series(y_test, name='Actual'),
        pd.Series(y_pred_test, name='Predicted')
    )

    return metrics_train, metrics_test, cm_train, cm_test