import numpy as np
import cv2 as cv
import pandas as pd
import os
import pickle
import tensorflow as tf
import sklearn.preprocessing
import src.feature_engineering as feature_engineering


class Classifier:
    """
    Class to hold a classifier.
    """

    def __init__(self, load_dir, configurations):
        """
        Constructor
        """

        # Initialize variables
        self.dst_universal_mass = 0
        self.dst_unscaled_masses = np.array([])
        self.normalizer = sklearn.preprocessing.StandardScaler()

        # Save the configurations for later use by other member functions
        self.config = configurations

        # Initialize a PCA projector instance if needed
        if self.config['classifier_type'] == 'SVM' and 'HOG' in self.config["svm_feature_types"]:
            self.pca_projector = feature_engineering.PCAProjector(load_dir=load_dir)

        # Load data normalizer
        self.load_normalizer(load_dir, configurations['classifier_type'])

        # Set the classification function to use
        if configurations['classifier_type'] == 'NN':
            self.classify_fn = self.convolutional_neural_network
        elif configurations['classifier_type'] == 'RF':
            self.classify_fn = self.random_forest
        elif configurations['classifier_type'] == 'SVM':
            self.classify_fn = self.support_vector_machine
        else:
            raise Exception('Unknown classifier type ' + configurations['classifier_type'] + '.')

        # Read the object labels from file
        labels_csv = pd.read_csv('./model/labels.csv')
        labels_index = list(labels_csv.columns).index('Labels')
        self.labels = np.array(labels_csv)[labels_index]

        # Load the classifier model and if necessary feature descriptor model
        if configurations['classifier_type'] == 'NN':
            if self.config['nn_network_architecture'] == 'resnet101':
                self.model = tf.keras.models.load_model(os.path.join(load_dir, 'Resnet.h5'))
            else:
                self.model = tf.keras.models.load_model(os.path.join(load_dir, 'CNN_' + str(self.config['nn_network_architecture']) + '.h5'))

        elif configurations['classifier_type'] == 'RF':
            self.feature_extractor = feature_engineering.KeypointFeaturesExtractor(load_dir, configurations)
            with open(os.path.join(load_dir, 'RF.pkl'), 'rb') as model_file:
                self.model = pickle.load(model_file)

        elif configurations['classifier_type'] == 'SVM':
            with open(os.path.join(load_dir, 'SVM.pkl'), 'rb') as model_file:
                self.model = pickle.load(model_file)

    def classify(self, image):
        """
        Classify an image.
        :param image: Input image to classify
        :return: Class probabilities
        """

        # Resize the image to the specified size
        image = cv.resize(image, self.config['image_fix_size'])

        # Classify the image
        probability_vector = self.classify_fn(image)

        # If Dempster-Shafer fusion enabled, separate the Universal class value and keep the sum of probabilities at 1
        if self.config['decision_fusion_type'] == 'Dempster-Shafer':
            self.dst_unscaled_masses = probability_vector[: -1]
            self.dst_universal_mass = probability_vector[-1]
            probability_vector = (1.0 / np.sum(self.dst_unscaled_masses)) * self.dst_unscaled_masses

        return probability_vector

    def convolutional_neural_network(self, image):
        """
        Classify an image using convolutional neural network.
        :param image: Input image to classify
        :return: Class probability (1D array)
        """

        # Resize the image to fit the network input
        image = cv.resize(image, self.model.layers[0].input_shape[0][2:0:-1])

        # Convert pixel values to the desired range
        if self.config['nn_network_architecture'] == 'resnet101':
            image = tf.keras.applications.resnet.preprocess_input(image)
        else:
            image = image / 255.0

        # Add the samples axis
        image = np.expand_dims(image, axis=0)

        # Classify
        class_scores = np.squeeze(self.model.predict(image))

        return class_scores

    def support_vector_machine(self, image):
        """
        Classify an image using support vector machine.
        :param image: Input image to classify
        :return: Class probability (1D array)
        """

        # Extract features
        hog_features, color_hist_features, hu_moments_features = feature_engineering.extract_engineered_features(image,
                                                                                                                 feature_types=self.config["svm_feature_types"],
                                                                                                                 hog_window_size=self.config["hog_window_size"],
                                                                                                                 hog_block_size=self.config["hog_block_size"],
                                                                                                                 hog_block_stride=self.config["hog_block_stride"],
                                                                                                                 hog_cell_size=self.config["hog_cell_size"],
                                                                                                                 hog_bin_no=self.config["hog_bin_no"],
                                                                                                                 color_histogram_size=self.config["color_histogram_size"])

        # Reduce HOG features
        if 'HOG' in self.config["svm_feature_types"]:
            hog_features = self.pca_projector.pca_project(sample=hog_features)

        # Concatenate the feature vectors
        feature_vector = np.concatenate((hog_features, color_hist_features, hu_moments_features))

        # Normalize the input feature vector
        feature_vector = self.normalizer.transform(feature_vector.reshape((1, -1)))

        # Classify
        class_scores = np.squeeze(self.model.predict_proba(feature_vector))

        return class_scores

    def random_forest(self, image):
        """
        Classify an image using random forest.
        :param image: Input image to classify
        :return: Class probability (1D array)
        """

        # Extract features
        feature_vector = self.feature_extractor.extract_features(image)

        # Normalize the input feature vector
        feature_vector = self.normalizer.transform(feature_vector.reshape((1, -1)))

        # Classify
        class_scores = np.squeeze(self.model.predict_proba(feature_vector))

        return class_scores

    def load_normalizer(self, load_dir, classifier_type):
        """
        :param load_dir: Load directory (str)
        :param classifier_type: Classifier type to load the respective feature normalizer
        Load data normalizer from drive.
        """

        # Set the name of the normalizer to load
        if classifier_type == 'RF':
            filename = 'Normalizer_RF'
        elif classifier_type == 'SVM':
            filename = 'Normalizer_SVM'
        else:   # In the case of any classifier that does not need a normalizer to load, just skip
            return

        # Load normalizer from file
        with open(os.path.join(load_dir, filename + '.pkl'), 'rb') as normalizer_file:
            self.normalizer = pickle.load(normalizer_file)

    def find_winner_class(self, probability_vector):
        """
        Function to find the winner class in a computed probability vector.
        :param probability_vector: Input probability vector (1D array)
        :return: Winner label (str), winner label code (int), winner probability (float)
        """

        # Find maximum location
        winner_code = np.argmax(probability_vector)

        # Find the winner label and probability
        winner_label = self.labels[winner_code]
        winner_probability = probability_vector[winner_code]

        return winner_label, winner_code, winner_probability

    def get_dst_masses(self):
        """
        Getter function for Dempster-Shafer fusion's mass.
        :return: Unscaled class masses of objects, Universal class mass
        """

        return self.dst_unscaled_masses, self.dst_universal_mass
