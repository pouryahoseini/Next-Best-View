import numpy as np
import pickle
import cv2 as cv
from sklearn.decomposition import PCA


def extract_features(images, feature_types, hog_window_size, hog_block_size, hog_block_stride, hog_cell_size, hog_bin_no, color_histogram_size):
    """
    Extract features from each image in the input.
    :param images: Input images (4D array: image_no, row, column, channel; or 3D array: row, column, channel)
    :param feature_types: List of feature type to use (elements of the list can be: 'HOG', 'HuMoments', 'ColorHistogram')
    :param hog_window_size: HOG window size
    :param hog_block_size: HOG block size
    :param hog_block_stride: HOG block stride
    :param hog_cell_size: HOG cell size
    :param hog_bin_no: HOG bin number per cell
    :param color_histogram_size: Number of color histogram features
    :return: HOG features, color histogram features, Hu moments features (if input is 4D, each output is 2D; if input is 3D, each input is 1D)
    """

    # If the input is 3D, reshape it to 4D
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    # Initialize the output feature vectors
    hog_features = np.zeros(images.shape[0], dtype=object)
    color_hist_features = np.zeros(images.shape[0], dtype=object)
    hu_moments_features = np.zeros(images.shape[0], dtype=object)

    # Repeat for all the samples
    for image_no, image in enumerate(images):

        # Extract HOG features
        if 'HOG' in feature_types:
            hog_features[image_no] = hog_descriptor(image, hog_window_size, hog_block_size, hog_block_stride, hog_cell_size, hog_bin_no)

        # Extract color histogram features
        if 'ColorHistogram' in feature_types:
            color_hist_features[image_no] = color_histogram_descriptor(image, color_histogram_size)

        # Extract Hu moments
        if 'HuMoments' in feature_types:
            hu_moments_features[image_no] = hu_moments(image)

    # Merge the array of arrays into a 2D array, if needed
    if 'HOG' in feature_types:
        hog_features = np.squeeze(np.concatenate([np.expand_dims(row, axis=0) for row in hog_features], axis=0))
    else:
        hog_features = np.zeros((images.shape[0], 0), dtype=float)
    if 'ColorHistogram' in feature_types:
        color_hist_features = np.squeeze(np.concatenate([np.expand_dims(row, axis=0) for row in color_hist_features], axis=0))
    else:
        color_hist_features = np.zeros((images.shape[0], 0), dtype=float)
    if 'HuMoments' in feature_types:
        hu_moments_features = np.squeeze(np.concatenate([np.expand_dims(row, axis=0) for row in hu_moments_features], axis=0))
    else:
        hu_moments_features = np.zeros((images.shape[0], 0), dtype=float)

    return hog_features, color_hist_features, hu_moments_features


def hu_moments(image):
    """
    Extract Hu moments of an image.
    :param image: Input image (3D array)
    :return: Hu moments feature vector (1D array)
    """

    # Compute moments for each channel
    moments_b = cv.moments(image[:, :, 0])
    moments_g = cv.moments(image[:, :, 1])
    moments_r = cv.moments(image[:, :, 2])

    # Compute Hu moments
    hu_moments_b = np.squeeze(cv.HuMoments(moments_b))
    hu_moments_g = np.squeeze(cv.HuMoments(moments_g))
    hu_moments_r = np.squeeze(cv.HuMoments(moments_r))

    # Concatenate the feature vectors
    hu_moments_features = np.concatenate((hu_moments_b, hu_moments_g, hu_moments_r), axis=0)

    return hu_moments_features


def color_histogram_descriptor(image, color_histogram_size):
    """
    Extract 2D color histogram features of an image (output is 1D).
    :param image: Input image (3D array)
    :param color_histogram_size: Number of desired features
    :return: Output color histogram feature vector (1D array)
    """

    # Determine number of bins in each axis of the 2D histograms
    n_bins = int(np.sqrt(color_histogram_size / 2.0))

    # Transform the image to Luv color space
    luv_image = cv.cvtColor(image, cv.COLOR_BGR2Luv)

    # Calculate 2D histogram of Luv color space (u and v channels)
    uv_hist = cv.calcHist(luv_image, channels=(1, 2), histSize=(n_bins, n_bins), ranges=((0, 256), (0, 256)), mask=None)

    # Transform the input image to HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Calculate 2D histogram of HSV color space (H and S channels)
    hs_hist = cv.calcHist(hsv_image, channels=(0, 1), histSize=(n_bins, n_bins), ranges=((0, 180), (0, 256)), mask=None)

    # Flatten the 2D histograms
    uv_hist = uv_hist.ravel()
    hs_hist = hs_hist.ravel()

    # Concatenate the histogram feature vectors
    color_hist_features = np.concatenate((uv_hist, hs_hist), axis=0)

    # Normalize the feature vector
    color_hist_features = cv.normalize(color_hist_features, None, alpha=1, beta=0, norm_type=cv.NORM_L2, dtype=np.float)

    return color_hist_features


def hog_descriptor(image, hog_window_size, hog_block_size, hog_block_stride, hog_cell_size, hog_bin_no):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    :param image: Input color image (3D array)
    :param hog_window_size: HOG window size
    :param hog_block_size: HOG block size
    :param hog_block_stride: HOG block stride
    :param hog_cell_size: HOG cell size
    :param hog_bin_no: HOG bin number per cell
    :return: HOG feature vector
    """

    # Define a Histogram of Oriented Gradients (HOG) descriptor
    hog = cv.HOGDescriptor(winSize=hog_window_size, blockSize=hog_block_size, blockStride=hog_block_stride, cellSize=hog_cell_size, nbins=hog_bin_no)

    # Convert to grayscale image
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Image is resized first to the HOG window size, thus no sliding on the image
    resized_gray_image = cv.resize(gray_image, hog_window_size)

    # Extract HOG features
    hog_features = hog.compute(resized_gray_image, winStride=None, padding=(0, 0))

    # Normalize HOG features
    hog_features = cv.normalize(hog_features, None, alpha=1, beta=0, norm_type=cv.NORM_L2, dtype=np.float)

    return hog_features


def pca_train(features_dataset, number_of_features):
    """
    PCA training.
    :param features_dataset: Input data to train the PCA on it (2D array)
    :param number_of_features: Maximum number of features in the reduced feature vector
    :return: Trained PCA object (scikit-learn)
    """

    # Create a PCA object
    pca = PCA(n_components=number_of_features)

    # Train the PCA
    pca.fit(features_dataset)

    # Save the trained PCA object on drive
    with open('./model/PCA.pkl', 'wb') as pca_file:
        pickle.dump(pca, pca_file)

    return pca


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

