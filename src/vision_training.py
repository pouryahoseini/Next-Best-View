import numpy as np
import cv2 as cv
import pandas as pd
import os
import pickle
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.ensemble
import tensorflow as tf
import src.train_data_augmentation as train_augmentation
import src.feature_engineering as feature_engineering


def train_vision(configurations):
    """
    Train vision model.
    :return: trained model
    """

    # Set the vision training root address
    root_address = './vision_training/'

    # Read the dataset from files on drive
    dataset_images, dataset_labels, label_codes = get_train_samples(root_address=root_address,
                                                                    image_size=configurations["train_image_fix_size"],
                                                                    train_images_extension=configurations["train_images_extension"])

    # Data augmentation
    augmented_dataset_images, augmented_dataset_labels = train_augmentation.train_data_augmentation(dataset_images, dataset_labels)

    # In the case of Dempster-Shafer fusion, add the 'Universal' class to the dataset
    if configurations["decision_fusion_type"] == 'DST':
        dataset_images, dataset_labels, label_codes = create_dempster_shafer_dataset(dataset_images, dataset_labels, label_codes,
                                                                                     augmented_dataset_images=augmented_dataset_images,
                                                                                     augmented_dataset_labels=augmented_dataset_labels,
                                                                                     universal_class_ratio_to_dataset=configurations["dst_universal_class_ratio_to_dataset"],
                                                                                     dst_augment_universal_class=configurations["dst_augment_universal_class"])
    else:
        dataset_images = augmented_dataset_images
        dataset_labels = augmented_dataset_labels

    # Shuffle the data
    shuffled_indices = np.choice(dataset_labels.shape[0], size=dataset_labels.shape[0], replace=False)
    dataset_images = dataset_images[shuffled_indices]
    dataset_labels = dataset_labels[shuffled_indices]

    # Save the label codes on drive
    df = pd.DataFrame({'Codes': list(range(label_codes.shape[0])), 'Labels': label_codes})
    df.to_csv(os.path.join(root_address, 'labels.csv'))

    # Choose the learning model
    if configurations["classifier_type"] == 'SVM':

        # Extract features
        hog_features, color_hist_features, hu_moments_features = feature_engineering.extract_features(dataset_images,
                                                                                                      feature_types=configurations["svm_feature_types"],
                                                                                                      hog_window_size=configurations["hog_window_size"],
                                                                                                      hog_block_size=configurations["hog_block_size"],
                                                                                                      hog_block_stride=configurations["hog_block_stride"],
                                                                                                      hog_cell_size=configurations["hog_cell_size"],
                                                                                                      hog_bin_no=configurations["hog_bin_no"],
                                                                                                      color_histogram_size=configurations["color_histogram_size"])

        if 'HOG' in configurations["svm_feature_types"]:
            # Train PCA feature reduction
            pca = feature_engineering.pca_train(features_dataset=hog_features, number_of_features=configurations["hog_reduced_features_no"])

            # Reduce HOG features
            hog_features = feature_engineering.pca_project(sample=hog_features, pca=pca)

        # Concatenate the feature vectors
        feature_vector = np.concatenate((hog_features, color_hist_features, hu_moments_features), axis=1)

        # Train SVM
        model = SupportVectorMachine(feature_dataset=feature_vector,
                                     label_dataset=dataset_labels,
                                     save_directory='./model/',
                                     svm_kernel=configurations["svm_kernel"])

    elif configurations["classifier_type"] == 'RF':

    elif configurations["classifier_type"] == 'NN':

    else:
        raise Exception("Classifier type ' + classifier_type + ' not recognized.")


def SupportVectorMachine(feature_dataset, label_dataset, save_directory, svm_kernel):
    """
    Train a support vector machine classifier.
    :param feature_dataset: Input feature dataset (2D array: sample_no, features)
    :param label_dataset: Input label dataset (1D array)
    :param save_directory: Directory to save the trained model (str)
    :param svm_kernel: SVM kernel to use (str)
    :return: The trained model
    """

    # Normalize the input feature vector
    feature_dataset = sklearn.preprocessing.normalize(feature_dataset, norm='l2', axis=1)

    # Make the label vector a 1D array by unraveling
    label_dataset = label_dataset.ravel()

    # Set the SVM classifier configurations
    cache_size = 10000
    class_weight = 'balanced'

    # Set cross-validation settings
    c_range = np.logspace(0, 1, 3)
    gamma_range = np.logspace(0, 1, 3)
    param_grid = dict(gamma=gamma_range, C=c_range)
    cross_validation_settings = sklearn.model_selection.KFold(n_splits=5, shuffle=True)

    # Find the optimal classifier parameters (C and Gamma)
    svm_to_be_optimized = sklearn.svm.SVC(probability=True, cache_size=cache_size, class_weight=class_weight, decision_function_shape='ovr', kernel=svm_kernel)
    grid_of_classifiers = sklearn.model_selection.GridSearchCV(svm_to_be_optimized, param_grid=param_grid, scoring=['accuracy', 'recall_macro', 'precision_macro', 'neg_log_loss'], refit='neg_log_loss', cv=cross_validation_settings, n_jobs=-1, verbose=3)

    grid_of_classifiers.fit(feature_dataset, label_dataset)

    # Define the SVM classifier
    svm_classifier = sklearn.svm.SVC(probability=True, cache_size=cache_size, class_weight=class_weight, decision_function_shape='ovr', kernel=svm_kernel, **grid_of_classifiers.best_params_)
    svm_classifier.fit(feature_dataset, label_dataset)

    # Print the best found parameters and the best score
    print('\n\nBest Accuracy: ' + str(grid_of_classifiers.best_score_))
    print('Best Parameters: \n{}\n\n'.format(grid_of_classifiers.best_params_))

    # Save the trained classifier
    file_address = os.path.join(save_directory, 'SVM.pkl')
    with open(file_address, "wb") as svm_file:
        pickle.dump(svm_classifier, svm_file)

    return svm_classifier


def create_dempster_shafer_dataset(dataset_images, dataset_labels, label_codes, augmented_dataset_images, augmented_dataset_labels, universal_class_ratio_to_dataset, dst_augment_universal_class):
    """
    Create a dataset with an added 'Universal' class, equal to the specified portion of the training samples, for Dempster-Shafer fusion.
    :param dataset_images: Input dataset images (4D array)
    :param dataset_labels: Input dataset labels (1D array)
    :param label_codes: Class label names (1D array of str)
    :param augmented_dataset_images: Augmented dataset images
    :param augmented_dataset_labels: Augmented dataset labels
    :param universal_class_ratio_to_dataset: Ratio of the number of samples in the 'Universal' class to the total number of samples in the dataset
    :param dst_augment_universal_class: Extract the Universal class from the augmented data (boolean)
    :return: updated dataset images, labels, label codes
    """

    # Check if the number of sample images and labels are equal
    assert dataset_images.shape[0] == dataset_labels.shape[0], "In creating Dempster-Shafer dataset: Number of sample images and labels are not equal."

    if dst_augment_universal_class:
        # Shuffle the dataset
        shuffled_indices = np.random.choice(augmented_dataset_labels.shape[0], size=augmented_dataset_labels.shape[0], replace=False)
        dst_dataset_images = augmented_dataset_labels[shuffled_indices]

        # Decide on the number of samples in the universal class
        universal_class_sample_no = int(universal_class_ratio_to_dataset * augmented_dataset_labels.shape[0])
    else:
        # Shuffle the dataset
        shuffled_indices = np.random.choice(dataset_labels.shape[0], size=dataset_labels.shape[0], replace=False)
        dst_dataset_images = dataset_images[shuffled_indices]

        # Decide on the number of samples in the universal class
        universal_class_sample_no = int(universal_class_ratio_to_dataset * dataset_labels.shape[0])

    # Add a portion of the dataset to it
    dst_dataset_images = np.concatenate((augmented_dataset_images, dst_dataset_images[: universal_class_sample_no]), axis=0)
    dst_dataset_labels = np.concatenate((augmented_dataset_labels, np.ones(universal_class_sample_no, dtype=augmented_dataset_labels.dtype) * label_codes.shape[0]), axis=0)

    # Add the universal class label code
    dst_label_codes = np.append(label_codes, 'Universal')

    return dst_dataset_images, dst_dataset_labels, dst_label_codes


def get_train_samples(root_address, image_size, train_images_extension):
    """
    Read all the train images and save in a numpy array.
    :param root_address: Parent directory
    :param image_size: Image size for each sample in the constructed dataset (a tuple with 2 elements)
    :param train_images_extension: Extension of image files in the train directories (str)
    :return: The constructed dataset and labels (numpy arrays of 4D and 1D shape), label code (1D numpy array)
    """

    # List the directories in the root address
    list_of_folders = [folder for folder in os.listdir(root_address) if os.path.isdir(folder)]

    # Initialize the dataset
    dataset_labels = np.zeros(1000, dtype=int)
    dataset_images = np.zeros((1000,) + image_size + (3,), dtype=np.uint8)
    label_code = np.array([])

    # Construct the dataset
    sample_no = 0
    class_no = -1
    for object_class in list_of_folders:

        # Get the current address
        current_address = os.path.join(root_address, object_class)

        # Get the list of images in the current folder
        current_image_list = [image_name for image_name in os.listdir(current_address) if (image_name[-(len(train_images_extension) + 1)] == '.' + train_images_extension)]

        # Increment the object number if there is any train image here
        if len(current_image_list) != 0:
            class_no += 1
            label_code = np.append(label_code, object_class)

        # Check if the dataset size has to be increased
        if dataset_labels.shape[0] < (sample_no + len(current_image_list)):
            dataset_labels = np.append(dataset_labels, np.zeros(1000 + len(current_image_list), dtype=dataset_labels.dtype), axis=0)
            dataset_images = np.append(dataset_images, np.zeros((1000 + len(current_image_list),) + dataset_images[1:], dtype=dataset_images.dtype), axis=0)

        # Read images in the current folder and resize
        image_no = -1
        for image_no, image in enumerate(current_image_list):
            dataset_labels[sample_no + image_no] = class_no
            dataset_images[sample_no + image_no] = cv.resize(cv.imread(os.path.join(current_address, image), cv.IMREAD_COLOR), dsize=image_size)

        # Update the number of samples written so far
        sample_no += image_no + 1

    # Remove unused elements in the dataset arrays
    dataset_labels = dataset_labels[: sample_no]
    dataset_images = dataset_images[: sample_no]

    # Print number of read samples
    print(str(sample_no) + ' training images read.')

    return dataset_images, dataset_labels, label_code
