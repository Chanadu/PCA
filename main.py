# importing required libraries
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Orthonormalizes a set of vectors using the Gram-Schmidt process.

    Args:
        vectors (numpy.ndarray): A 2D numpy array where each column represents a vector.

    Returns:
        numpy.ndarray: A 2D numpy array containing the orthonormalized vectors as columns.
    """
    n, m = vectors.shape

    if n < m:
        raise ValueError(
            "Number of vectors must be less than or equal to the dimension")

    orthonormal_vectors = np.zeros((n, m))

    for i in range(m):
        temp_vector = vectors[:, i]
        for j in range(i):
            projection = np.dot(orthonormal_vectors[:, j], vectors[:, i])
            temp_vector = temp_vector - projection * orthonormal_vectors[:, j]

        norm = np.linalg.norm(temp_vector)

        if norm < 1e-10:
            raise ValueError("Vectors are linearly dependent")

        orthonormal_vectors[:, i] = temp_vector / norm

    return orthonormal_vectors


class MyPCA:

    def __init__(self, num_components: int):
        self.num_components = num_components
        # self.components = None
        # self.mean = None
        # self.variance_share = None

    def fit(self, data: np.ndarray):
        # Standardize data
        data = data.copy()

        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0)
        data_std: np.ndarray = (data - self.mean) / self.scale

        # data_std = gram_schmidt(data)

        # Creates the covariance matrix
        cov_mat = np.cov(data_std.T)

        # Computes the eignenvalues and eigenvectors of the covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Sort the eigenvalues and eigenvectors
        sort_idx = np.argsort(eig_vals)[::-1]
        values = eig_vals[sort_idx]
        vectors = eig_vecs[:, sort_idx]

        # Select the top k eigenvalues and eigenvectors
        self.components = vectors[:self.num_components]
        self.variance_share = np.sum(
            values[:self.num_components]) / np.sum(values)
        self.explained_variance_ratio = values[:
                                               self.num_components] / np.sum(values)
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, data):
        data = data.copy()
        data_std = (data - self.mean) / self.scale

        # data_proj = data_std.dot(self.components.T)

        return np.dot(data_std, self.components.T)


def read_csv_file(dataset1_paths: list, dataset2_paths: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a CSV file and returns the data as a numpy array.

    Args:
            file_path (str): Path to the CSV file.

    Returns:
            numpy.ndarray: Data from the CSV file.
    """
    combined_df = pd.DataFrame()
    labels = []

    for file_path in dataset1_paths:
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df = df.dropna(axis=1)  # Drop columns with NaN values
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        labels.extend([0] * len(df))  # Label rows from the first dataset as 0

    for file_path in dataset2_paths:
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df = df.dropna(axis=1)  # Drop columns with NaN values
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        labels.extend([1] * len(df))  # Label rows from the second dataset as 1

    # df = pd.read_csv(file_path, header=None)
    # df = pd.read_csv(file_path, header=None, skiprows=1)
    # df = df.dropna(axis=1)

    print(combined_df.head())
    # Check the column names and indices
    print("Columns in the dataset:", combined_df.columns)
    # Check for missing or invalid values
    print("Any NaN values in the dataset:", combined_df.isnull().any().any())
    print("Any inf values in the dataset:", np.isinf(combined_df.values).any())

    return combined_df.values, np.array(labels)


def custom_pca(X: np.ndarray, dataset_labels: np.ndarray, n_components: int = 2):
    """
    Custom PCA implementation to visualize the data in 2D.
    Args:
                X (numpy.ndarray): Input data.
                dataset_labels (numpy.ndarray): Labels for the data points.
                n_components (int): Number of principal components to keep.
        """
    pca = MyPCA(num_components=2).fit(X)

    print('Components:\n', pca.components)
    print('Explained variance ratio from scratch:\n',
          pca.explained_variance_ratio)
    print('Cumulative explained variance from scratch:\n',
          pca.cum_explained_variance)

    X_pca = pca.transform(X)
    print('Transformed data shape from scratch:', X_pca.shape)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dataset_labels)
    plt.xlabel('PC1')
    plt.xticks([])
    plt.ylabel('PC2')
    plt.yticks([])
    plt.title('2 components, captures {}% of total variation'.format(
        (pca.cum_explained_variance[1] * 100).round(2)))
    # plt.colorbar(label='Class')
    plt.show()


def builtin_pca(X: np.ndarray, dataset_labels: np.ndarray):
    """
    Built-in PCA implementation to visualize the data in 2D.
    Args:
                X (numpy.ndarray): Input data.
                dataset_labels (numpy.ndarray): Labels for the data points.
    """
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2).fit(X_std)

    print('Components:\n', pca.components_)
    print('Explained variance ratio:\n', pca.explained_variance_ratio_)

    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print('Cumulative explained variance:\n', cum_explained_variance)

    X_pca = pca.transform(X_std)  # Apply dimensionality reduction to X.
    print('Transformed data shape:', X_pca.shape)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dataset_labels)
    plt.xlabel('PC1')
    plt.xticks([])
    plt.ylabel('PC2')
    plt.yticks([])
    plt.title('2 components, captures {}% of total variation'.format(
        cum_explained_variance[1].round(4)*100))
    plt.show()


dataset1_paths = [
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 1.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 2.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 3.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 4.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 5.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/ADHD/10/Emotiv/emo zeynep 6.csv',
]
dataset2_paths = [
    './FOCUSDataset/_To Share EEG Data/New EEG Data/Non-ADHD/1/Emotive/abdul_emo1.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/Non-ADHD/1/Emotive/abdul_emo2.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/Non-ADHD/1/Emotive/abdul_emo3.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/Non-ADHD/1/Emotive/abdul_emo4.csv',
    './FOCUSDataset/_To Share EEG Data/New EEG Data/Non-ADHD/1/Emotive/abdul_emo5.csv',
]

X, dataset_labels = read_csv_file(dataset1_paths, dataset2_paths)

# custom_pca(X, dataset_labels)
builtin_pca(X, dataset_labels)
