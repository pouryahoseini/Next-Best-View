import numpy as np
import cv2 as cv
import pandas as pd
import os
import pickle
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.ensemble
import sklearn.preprocessing
import tensorflow as tf
import src.train_data_augmentation as train_augmentation
import src.feature_engineering as feature_engineering


def train_vision(configurations):
    """
    Train vision model.
    :param configurations: Configurations dictionary
    """

    # Set the vision training root address
    root_address = './vision_training/'

    # Read the dataset from files on drive
    dataset_images, dataset_labels, label_codes = get_train_samples(root_address=root_address,
                                                                    image_size=configurations["image_fix_size"],
                                                                    train_images_extension=configurations["train_images_extension"])

    # Data augmentation
    augmented_dataset_images, augmented_dataset_labels = train_augmentation.train_data_augmentation(dataset_images, dataset_labels)

    # In the case of Dempster-Shafer fusion, add the 'Universal' class to the dataset
    if configurations["decision_fusion_type"] == 'Dempster-Shafer':
        dataset_images, dataset_labels, label_codes = create_dempster_shafer_dataset(dataset_images, dataset_labels, label_codes,
                                                                                     augmented_dataset_images=augmented_dataset_images,
                                                                                     augmented_dataset_labels=augmented_dataset_labels,
                                                                                     universal_class_ratio_to_dataset=configurations["dst_universal_class_ratio_to_dataset"],
                                                                                     dst_augment_universal_class=configurations["dst_augment_universal_class"])
    else:
        dataset_images = augmented_dataset_images
        dataset_labels = augmented_dataset_labels

    # Shuffle the data
    shuffled_indices = np.random.permutation(dataset_labels.shape[0])
    dataset_images = dataset_images[shuffled_indices]
    dataset_labels = dataset_labels[shuffled_indices]

    # Save the label codes on drive
    df = pd.DataFrame({'Codes': list(range(label_codes.shape[0])), 'Labels': label_codes})
    df.to_csv(os.path.join('./model/', 'labels.csv'))

    # Repeat for all tiles and the original images
    for tile_name, tile_no in [('original', 0)] + [('tile_' + str(tile_no), tile_no) for tile_no in range(1, 6)]:

        # Create save directory on drive if it does not exist
        if not os.path.exists(os.path.join('./model/', tile_name)):
            os.mkdir(os.path.join('./model/', tile_name))

        # Extract the tiles from the dataset
        dataset_tiles = extract_tile(dataset_images=dataset_images, tile_no=tile_no)

        # Train the learning model for the current tile
        train_model(dataset_images=dataset_tiles, dataset_labels=dataset_labels, configurations=configurations, save_directory='./model/' + tile_name)


def extract_tile(dataset_images, tile_no):
    """
    Extract tiles from the images in the input dataset.
    :param dataset_images: Input dataset images (4D array: sample_no, row, column, channel)
    :param tile_no: Tile number that defines where in each image to crop and extract (int)
    :return: Dataset of images with each image a cropping of the original image
    """

    # Extract based on the requested tile number
    if tile_no == 0:    # Original image
        extracted_tiles = dataset_images
    elif tile_no == 1:
        extracted_tiles = dataset_images[:, : int(dataset_images.shape[1] / 3), : int(dataset_images.shape[2] / 3)]
    elif tile_no == 2:
        extracted_tiles = dataset_images[:, : int(dataset_images.shape[1] / 3), int(dataset_images.shape[2] / 3): int(2 * dataset_images.shape[2] / 3)]
    elif tile_no == 3:
        extracted_tiles = dataset_images[:, : int(dataset_images.shape[1] / 3), int(2 * dataset_images.shape[2] / 3):]
    elif tile_no == 4:
        extracted_tiles = dataset_images[:, int(dataset_images.shape[1] / 3): int(2 * dataset_images.shape[1] / 3), : int(dataset_images.shape[2] / 3)]
    elif tile_no == 5:
        extracted_tiles = dataset_images[:, int(dataset_images.shape[1] / 3): int(2 * dataset_images.shape[1] / 3), int(2 * dataset_images.shape[2] / 3):]
    else:
        raise Exception("Error: Tile number " + str(tile_no) + " undefined.")

    return extracted_tiles


def train_model(dataset_images, dataset_labels, configurations, save_directory):
    """
    Train an image classifier.
    :param dataset_images: Input images (4D array: sample_no, row, column, channel)
    :param dataset_labels: Input labels (1D array)
    :param configurations: Configurations dictionary
    :param save_directory: Directory to save the trained model (str)
    :return: Trained model
    """

    # Choose the learning model
    if configurations["classifier_type"] == 'SVM':

        # Extract features
        hog_features, color_hist_features, hu_moments_features = feature_engineering.extract_engineered_features(dataset_images,
                                                                                                                 feature_types=configurations["svm_feature_types"],
                                                                                                                 hog_window_size=configurations["hog_window_size"],
                                                                                                                 hog_block_size=configurations["hog_block_size"],
                                                                                                                 hog_block_stride=configurations["hog_block_stride"],
                                                                                                                 hog_cell_size=configurations["hog_cell_size"],
                                                                                                                 hog_bin_no=configurations["hog_bin_no"],
                                                                                                                 color_histogram_size=configurations["color_histogram_size"])

        if 'HOG' in configurations["svm_feature_types"]:
            # Train PCA feature reduction
            feature_engineering.pca_train(features_dataset=hog_features, number_of_features=configurations["hog_reduced_features_no"], save_directory=save_directory)

            # Reduce HOG features
            pca_projector = feature_engineering.PCAProjector(load_dir=save_directory)
            hog_features = pca_projector.pca_project(sample=hog_features)

        # Concatenate the feature vectors
        dataset_features = np.concatenate((hog_features, color_hist_features, hu_moments_features), axis=1)

        # Train SVM
        model = support_vector_machine(feature_dataset=dataset_features,
                                       label_dataset=dataset_labels,
                                       save_directory=save_directory,
                                       svm_kernel=configurations["svm_kernel"],
                                       cross_validation_splits=configurations["cross_validation_splits"])

    elif configurations["classifier_type"] == 'RF':

        # Train bag of words feature extractor and return the extracted features
        dataset_features, dataset_labels = feature_engineering.train_keypoint_features_extractor(images=dataset_images,
                                                                                                 labels=dataset_labels,
                                                                                                 bag_of_words_feature_type=configurations["bag_of_words_feature_type"],
                                                                                                 save_dir=save_directory,
                                                                                                 sift_features_no=configurations["sift_features_no"],
                                                                                                 sift_octave_layers=configurations["sift_octave_layers"],
                                                                                                 sift_contrast_threshold=configurations["sift_contrast_threshold"],
                                                                                                 sift_edge_threshold=configurations["sift_edge_threshold"],
                                                                                                 sift_sigma=configurations["sift_sigma"],
                                                                                                 kaze_threshold=configurations["kaze_threshold"],
                                                                                                 kaze_octaves_no=configurations["kaze_octaves_no"],
                                                                                                 kaze_octave_layers=configurations["kaze_octave_layers"],
                                                                                                 bow_cluster_no=configurations["bag_of_words_cluster_no"])

        # Train random forest
        model = random_forest(feature_dataset=dataset_features,
                              label_dataset=dataset_labels,
                              save_directory=save_directory,
                              rf_criterion=configurations["rf_criterion"],
                              rf_estimators_no=configurations["rf_estimators_no"],
                              cross_validation_splits=configurations["cross_validation_splits"])

    elif configurations["classifier_type"] == 'NN':

        # Train the convolutional neural network
        if configurations["nn_network_architecture"] == 'resnet101':
            model = resnet101(image_dataset=dataset_images,
                              label_dataset=dataset_labels,
                              save_directory=save_directory,
                              nn_epochs=configurations["nn_epochs"],
                              nn_max_learning_rate=configurations["nn_max_learning_rate"],
                              nn_batch_size=configurations["nn_batch_size"],
                              nn_validation_split=configurations["nn_validation_split"],
                              nn_early_stopping_patience=configurations["nn_early_stopping_patience"])
        else:
            model = convolutional_neural_network(image_dataset=dataset_images,
                                                 label_dataset=dataset_labels,
                                                 network_type=configurations["nn_network_architecture"],
                                                 save_directory=save_directory,
                                                 nn_epochs=configurations["nn_epochs"],
                                                 nn_max_learning_rate=configurations["nn_max_learning_rate"],
                                                 nn_batch_size=configurations["nn_batch_size"],
                                                 nn_validation_split=configurations["nn_validation_split"],
                                                 nn_early_stopping_patience=configurations["nn_early_stopping_patience"])

    else:
        raise Exception("Classifier type " + configurations["classifier_type"] + " not recognized.")

    return model


def convolutional_neural_network(image_dataset, label_dataset, network_type, save_directory, nn_epochs, nn_max_learning_rate, nn_batch_size, nn_validation_split, nn_early_stopping_patience):
    """
    Train convolutional neural network.
    :param image_dataset: Input image dataset (4D array: sample_no, row, column, channel)
    :param label_dataset: Input label dataset (1D array)
    :param network_type: An integer between 1 and 3 (i.e. 1, 2, 3) to indicate which of the three available network structures to use
    :param save_directory: Directory to save the trained model on drive (str)
    :param nn_epochs: Number of epochs to train the neural network (int)
    :param nn_max_learning_rate: Maximum learning rate during training of the neural network (float)
    :param nn_batch_size: Mini-batch size during training (int)
    :param nn_validation_split: Validation split of the training data (float between 0 and 1)
    :param nn_early_stopping_patience: Patience in epochs of early stopping of training (int)
    :return: The trained convolutional neural network model
    """

    # Change range of pixel values to between 0 and 1
    image_dataset = image_dataset / 255.0

    # Create categorical one-hot labels for the network output
    label_dataset_one_hot = tf.keras.utils.to_categorical(label_dataset)

    # Shuffle data
    shuffled_indices = np.random.permutation(image_dataset.shape[0])
    image_dataset = image_dataset[shuffled_indices]
    label_dataset_one_hot = label_dataset_one_hot[shuffled_indices]

    # Define the input layer of the neural network
    input_layer = tf.keras.layers.Input(shape=image_dataset.shape[1:])

    # Decide on the network architecture
    if network_type == '1':

        # Network 1: Convolutional layers
        ln_0 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(input_layer)

        cnn_layer_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(ln_0)
        activation_1 = tf.keras.layers.ReLU()(cnn_layer_1)

        do_1 = tf.keras.layers.Dropout(rate=0.1)(activation_1)

        cnn_layer_2 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_1)
        activation_2 = tf.keras.layers.ReLU()(cnn_layer_2)

        max_pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(activation_2)

        do_2 = tf.keras.layers.Dropout(rate=0.1)(max_pooling_1)

        cnn_layer_3 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_2)
        activation_3 = tf.keras.layers.ReLU()(cnn_layer_3)

        do_3 = tf.keras.layers.Dropout(rate=0.1)(activation_3)

        cnn_layer_4 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_3)
        activation_4 = tf.keras.layers.ReLU()(cnn_layer_4)

        max_pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding="valid")(activation_4)

        # Network 1: Dense layers
        flattened = tf.keras.layers.Flatten()(max_pooling_2)

        do_4 = tf.keras.layers.Dropout(rate=0.1)(flattened)

        dense_1 = tf.keras.layers.Dense(units=800, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_4)
        activation_5 = tf.keras.layers.ReLU()(dense_1)

        do_5 = tf.keras.layers.Dropout(rate=0.1)(activation_5)

        dense_2 = tf.keras.layers.Dense(units=150, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_5)
        activation_6 = tf.keras.layers.ReLU()(dense_2)

        output_layer = tf.keras.layers.Dense(units=label_dataset_one_hot.shape[1], activation='softmax', kernel_initializer='glorot_normal', activity_regularizer=None, use_bias=True)(activation_6)

    elif network_type == '2':   # Difference with Net 1: tanh instead of relu, average pooling instead of max pooling

        # Network 2: Convolutional layers
        ln_0 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(input_layer)

        cnn_layer_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(ln_0)
        activation_1 = tf.keras.activations.tanh(cnn_layer_1)

        do_1 = tf.keras.layers.Dropout(rate=0.1)(activation_1)

        cnn_layer_2 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_1)
        activation_2 = tf.keras.activations.tanh(cnn_layer_2)

        max_pooling_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid")(activation_2)

        do_2 = tf.keras.layers.Dropout(rate=0.1)(max_pooling_1)

        cnn_layer_3 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_2)
        activation_3 = tf.keras.activations.tanh(cnn_layer_3)

        do_3 = tf.keras.layers.Dropout(rate=0.1)(activation_3)

        cnn_layer_4 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_3)
        activation_4 = tf.keras.activations.tanh(cnn_layer_4)

        max_pooling_2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=None, padding="valid")(activation_4)

        # Network 2: Dense layers
        flattened = tf.keras.layers.Flatten()(max_pooling_2)

        do_4 = tf.keras.layers.Dropout(rate=0.1)(flattened)

        dense_1 = tf.keras.layers.Dense(units=800, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_4)
        activation_5 = tf.keras.activations.tanh(dense_1)

        do_5 = tf.keras.layers.Dropout(rate=0.1)(activation_5)

        dense_2 = tf.keras.layers.Dense(units=150, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_5)
        activation_6 = tf.keras.activations.tanh(dense_2)

        output_layer = tf.keras.layers.Dense(units=label_dataset_one_hot.shape[1], activation='softmax', kernel_initializer='glorot_normal', activity_regularizer=None, use_bias=True)(activation_6)

    elif network_type == '3':   # Difference with Net 1: One less dropout, CNN, one max pooling layers at the end of the convolutional layers

        # Network 3: Convolutional layers
        ln_0 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(input_layer)

        cnn_layer_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(ln_0)
        activation_1 = tf.keras.layers.ReLU()(cnn_layer_1)

        do_1 = tf.keras.layers.Dropout(rate=0.1)(activation_1)

        cnn_layer_2 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_1)
        activation_2 = tf.keras.layers.ReLU()(cnn_layer_2)

        max_pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(activation_2)

        do_2 = tf.keras.layers.Dropout(rate=0.1)(max_pooling_1)

        cnn_layer_3 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, kernel_initializer='glorot_normal', activity_regularizer='l2', padding='valid', strides=(1, 1))(do_2)
        activation_3 = tf.keras.layers.ReLU()(cnn_layer_3)

        # Network 3: Dense layers
        flattened = tf.keras.layers.Flatten()(activation_3)

        do_3 = tf.keras.layers.Dropout(rate=0.1)(flattened)

        dense_1 = tf.keras.layers.Dense(units=800, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_3)
        activation_4 = tf.keras.layers.ReLU()(dense_1)

        do_4 = tf.keras.layers.Dropout(rate=0.1)(activation_4)

        dense_2 = tf.keras.layers.Dense(units=150, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=True)(do_4)
        activation_5 = tf.keras.layers.ReLU()(dense_2)

        output_layer = tf.keras.layers.Dense(units=label_dataset_one_hot.shape[1], activation='softmax', kernel_initializer='glorot_normal', activity_regularizer=None, use_bias=True)(activation_5)

    else:
        raise ValueError("The CNN network type " + network_type + " is not recognized.")

    # Set the file name for saving the trained classifier
    file_address = os.path.join(save_directory, 'CNN_' + str(network_type) + '.h5')

    # Define a learning rate scheduler
    def schedule(epoch, lr):
        if epoch < (nn_epochs / 4):
            return nn_max_learning_rate
        elif epoch < (nn_epochs / 2):
            return nn_max_learning_rate / 3.0
        elif epoch < (3 * nn_epochs / 4):
            return nn_max_learning_rate / 10.0
        else:
            return nn_max_learning_rate / 20.0
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

    # Construct the NN model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Define the learning optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=nn_max_learning_rate)

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Set model checkpoints so that the best model is saved
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_address, monitor="val_loss", save_best_only=True, save_weights_only=False)

    # Define early stopping conditions and returning the best model after finishing the training
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=nn_early_stopping_patience, restore_best_weights=True)

    # Train the NN model
    model.fit(image_dataset, label_dataset_one_hot, epochs=nn_epochs, batch_size=nn_batch_size, validation_split=nn_validation_split, shuffle=True, callbacks=[model_checkpoint, learning_rate_scheduler, early_stopping])

    return model


def resnet101(image_dataset, label_dataset, save_directory, nn_epochs, nn_max_learning_rate, nn_batch_size, nn_validation_split, nn_early_stopping_patience):
    """
    Train a Resnet 101 with transfer learning and frozen convolutional layers.
    :param image_dataset: Input image dataset (4D array: sample_no, row, column, channel)
    :param label_dataset: Input label dataset (1D array)
    :param save_directory: Directory to save the trained model on drive (str)
    :param nn_epochs: Number of epochs to train the neural network (int)
    :param nn_max_learning_rate: Maximum learning rate during training of the neural network (float)
    :param nn_batch_size: Mini-batch size during training (int)
    :param nn_validation_split: Validation split of the training data (float between 0 and 1)
    :param nn_early_stopping_patience: Patience in epochs of early stopping of training (int)
    :return: The trained convolutional neural network model
    """

    # Adjust the range of input values to the proper expected range of Resnet 101
    image_dataset = tf.keras.applications.resnet.preprocess_input(image_dataset)

    # Create categorical one-hot labels for the network output
    label_dataset_one_hot = tf.keras.utils.to_categorical(label_dataset)

    # Shuffle data
    shuffled_indices = np.random.permutation(image_dataset.shape[0])
    image_dataset = image_dataset[shuffled_indices]
    label_dataset_one_hot = label_dataset_one_hot[shuffled_indices]

    # Define the pre-trained convolutional layers of Resnet 101
    resnet = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=image_dataset.shape[1:])

    # Freeze the pre-trained convolutional layers
    for layer in resnet.layers:
        layer.trainable = False

    # Add the fully-connected layers on top of the Resnet convolutional layers
    average_pool = tf.keras.layers.GlobalAveragePooling2D()(resnet.layers[-1].output)
    flatten = tf.keras.layers.Flatten()(average_pool)
    dense_1 = tf.keras.layers.Dense(200, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(label_dataset_one_hot.shape[1], activation='softmax')(dense_1)

    # Set the file name for saving the trained classifier
    file_address = os.path.join(save_directory, 'Resnet.h5')

    # Define a learning rate scheduler
    def schedule(epoch, lr):
        if epoch < (nn_epochs / 4):
            return nn_max_learning_rate
        elif epoch < (nn_epochs / 2):
            return nn_max_learning_rate / 3.0
        elif epoch < (3 * nn_epochs / 4):
            return nn_max_learning_rate / 10.0
        else:
            return nn_max_learning_rate / 20.0
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

    # Construct the NN model
    model = tf.keras.Model(inputs=resnet.inputs, outputs=dense_2)

    # Define the learning optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=nn_max_learning_rate)

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Set model checkpoints so that the best model is saved
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_address, monitor="val_loss", save_best_only=True, save_weights_only=False)

    # Define early stopping conditions and returning the best model after finishing the training
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=nn_early_stopping_patience, restore_best_weights=True)

    # Train the NN model
    model.fit(image_dataset, label_dataset_one_hot, epochs=nn_epochs, batch_size=nn_batch_size, validation_split=nn_validation_split, shuffle=True, callbacks=[model_checkpoint, learning_rate_scheduler, early_stopping])

    return model


def random_forest(feature_dataset, label_dataset, save_directory, rf_criterion, rf_estimators_no, cross_validation_splits):
    """
    Train a Random Forest Classifier.
    :param feature_dataset: Input feature dataset (2D array: sample_no, features)
    :param label_dataset: Input label dataset (1D array)
    :param save_directory: Directory to save the trained model (str)
    :param rf_criterion: Criterion to split a tree in the random forest
    :param rf_estimators_no: Number of trees in the random forest
    :param cross_validation_splits: Number of splits for cross-validation (int)
    :return: The trained model
    """

    # Normalize the dataset
    feature_dataset = train_normalizer(feature_dataset, save_directory, 'Normalizer_RF')

    # Make the label vector a 1D array by unraveling
    label_dataset = label_dataset.ravel()

    # Set cross-validation settings
    cross_validation_settings = sklearn.model_selection.KFold(n_splits=cross_validation_splits, shuffle=True)

    # Define a random forest classifier instance
    rf_to_be_optimized = sklearn.ensemble.RandomForestClassifier(n_estimators=rf_estimators_no, criterion=rf_criterion, class_weight='balanced', n_jobs=-1)

    # Set grid search parameters
    param_grid = dict(max_depth=(None, 50, 100), min_samples_split=(5, 10), min_samples_leaf=(1, 3))
    refit_metric = 'neg_log_loss'
    grid_of_classifiers = sklearn.model_selection.GridSearchCV(rf_to_be_optimized, param_grid=param_grid, scoring=['accuracy', 'recall_macro', 'precision_macro', 'neg_log_loss'], refit=refit_metric, cv=cross_validation_settings, n_jobs=-1, verbose=3)

    # Perform grid search to find the best parameters for the random forest classifier
    grid_of_classifiers.fit(feature_dataset, label_dataset)

    # Keep the best estimator
    rf_classifier = grid_of_classifiers.best_estimator_

    # Print the best found parameters and the best score
    print('\n\nBest ' + refit_metric + ': ' + str(grid_of_classifiers.best_score_))
    print('Best Parameters: {}\n'.format(grid_of_classifiers.best_params_))

    # Save the trained classifier
    file_address = os.path.join(save_directory, 'RF.pkl')
    with open(file_address, "wb") as rf_file:
        pickle.dump(rf_classifier, rf_file)

    return rf_classifier


def support_vector_machine(feature_dataset, label_dataset, save_directory, svm_kernel, cross_validation_splits):
    """
    Train a support vector machine classifier.
    :param feature_dataset: Input feature dataset (2D array: sample_no, features)
    :param label_dataset: Input label dataset (1D array)
    :param save_directory: Directory to save the trained model (str)
    :param svm_kernel: SVM kernel to use (str)
    :param cross_validation_splits: Number of splits for cross-validation (int)
    :return: The trained model
    """

    # Normalize the dataset
    feature_dataset = train_normalizer(feature_dataset, save_directory, 'Normalizer_SVM')

    # Make the label vector a 1D array by unraveling
    label_dataset = label_dataset.ravel()

    # Set the SVM classifier configurations
    cache_size = 10000
    class_weight = 'balanced'

    # Set cross-validation settings
    cross_validation_settings = sklearn.model_selection.KFold(n_splits=cross_validation_splits, shuffle=True)

    # Define a support vector machine classifier instance
    svm_to_be_optimized = sklearn.svm.SVC(probability=True, cache_size=cache_size, class_weight=class_weight, decision_function_shape='ovr', kernel=svm_kernel)

    # Set grid search parameters
    c_range = np.logspace(-1, 1, 3)
    kernels = ['poly', 'rbf']
    param_grid = dict(kernel=kernels, C=c_range)
    refit_metric = 'neg_log_loss'
    grid_of_classifiers = sklearn.model_selection.GridSearchCV(svm_to_be_optimized, param_grid=param_grid, scoring=['accuracy', 'recall_macro', 'precision_macro', 'neg_log_loss'], refit=refit_metric, cv=cross_validation_settings, n_jobs=-1, verbose=3)

    # Find the optimal classifier parameters (C and Gamma)
    grid_of_classifiers.fit(feature_dataset, label_dataset)

    # Keep the best estimator
    svm_classifier = grid_of_classifiers.best_estimator_

    # Print the best found parameters and the best score
    print('\n\nBest ' + refit_metric + ': ' + str(grid_of_classifiers.best_score_))
    print('Best Parameters: {}\n'.format(grid_of_classifiers.best_params_))

    # Save the trained classifier
    file_address = os.path.join(save_directory, 'SVM.pkl')
    with open(file_address, "wb") as svm_file:
        pickle.dump(svm_classifier, svm_file)

    return svm_classifier


def train_normalizer(dataset, save_directory, save_name=None):
    """
    Train a dataset normalizer, save it on drive, and return the normalized dataset.
    :param dataset: Input dataset (2D array: sample_no, features)
    :param save_directory: Directory to save the normalizer (str)
    :param save_name: Name of the saved normalizer on drive (str)
    :return: Normalized dataset
    """

    # Define a normalizer (scaler) instance
    normalizer = sklearn.preprocessing.StandardScaler()

    # Fit the normalizer and transform the dataset
    normalized_dataset = normalizer.fit_transform(dataset)

    # Save the normalizer on disk
    if save_name is None:
        save_name = 'Normalizer'
    with open(os.path.join(save_directory, save_name + '.pkl'), 'wb') as normalizer_file:
        pickle.dump(normalizer, normalizer_file)

    return normalized_dataset


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
    # dst_label_codes = np.append(label_codes, 'Universal')
    dst_label_codes = label_codes

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
    list_of_folders = [folder for folder in os.listdir(root_address) if os.path.isdir(os.path.join(root_address, folder))]

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
        current_image_list = [image_name for image_name in os.listdir(current_address) if (image_name[-(len(train_images_extension) + 1):] == '.' + train_images_extension)]

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
