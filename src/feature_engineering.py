import numpy as np
import pickle
from sklearn.decomposition import PCA


def pca_train(dataset, number_of_features):
    """
    PCA training.
    :param dataset: Input dataset to train the PCA on it
    :param number_of_features: Maximum number of features in the reduced feature vector
    :return: Trained PCA object (scikit-learn)
    """

    # Create a PCA object
    pca = PCA(n_components=number_of_features)

    # Train the PCA
    pca.fit(dataset)

    # Save the trained PCA object on drive
    with open('./model/PCA.pkl', 'wb') as pca_file:
        pickle.dump(pca, pca_file)


def pca_project(sample, pca):
    """
    Reduce features using a pre-trained PCA.
    :param sample: Sample or samples to reduce their features (1D or 2D numpy arrays)
    :param pca: Pre-trained PCA
    :return: Sample or samples with reduced features
    """

    # Check the shape of the sample array
    if sample.ndim == 1:
        sample = np.reshape(sample, (1, -1))
    elif sample.ndim > 2:
        raise Exception("Number of dimensions of the input data for PCA should be at most 2.")

    # Project to lower dimensions
    projected_sample = pca.transform(sample)

    # If the original sample was 1D, return 1D
    if sample.ndim == 1:
        projected_sample = np.squeeze(projected_sample)

    return projected_sample

