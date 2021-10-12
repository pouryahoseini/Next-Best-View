import configparser
import numpy as np
import os
import ast
import pandas as pd
import cv2 as cv
import src.vision_training as vision_training
import src.classification as classification
import src.test_data_augmentation as test_augmentation
import src.surface_normal as surface_normal
import src.fusion as fusion
from src import evaluation


class NextBestView:

    def __init__(self):
        """
        Constructor
        """

        # Initialize variables
        self.config = {}

        # Read configurations
        self.read_config('config.cfg')

        # Set some variables
        self.variations = 30
        self.total_tiles_no = 5

    def generate_scores(self):
        """
        Generate class probabilities of frontal and side images, and generate the scores of each tile in the frontal image.
        """

        # Set number of benchmarks, test variations per benchmark, and tiles
        benchmarks_no = len([directory for directory in os.listdir('./test_set/') if os.path.isdir(os.path.join('./test_set/', directory))])

        # Initialize variables
        tile_probabilities_specific = np.zeros(self.total_tiles_no, dtype=object)
        tile_probabilities_general = np.zeros(self.total_tiles_no, dtype=object)
        side_probabilities = np.zeros(self.total_tiles_no, dtype=object)
        fused_probabilities = np.zeros(self.total_tiles_no, dtype=object)
        surface_normal_score = np.zeros(self.total_tiles_no, dtype=np.float)
        hist_variance = np.zeros(self.total_tiles_no, dtype=np.float)
        hist_uniformity = np.zeros(self.total_tiles_no, dtype=np.float)
        hist_negative_entropy = np.zeros(self.total_tiles_no, dtype=np.float)
        csv_data = np.zeros((benchmarks_no * self.variations, 11), dtype=object)

        # Read index of object labels
        labels_data_frame = pd.read_csv('./model/labels.csv')
        label_names = labels_data_frame.to_numpy()[:, labels_data_frame.columns.to_list().index('Labels')]
        label_codes = labels_data_frame.to_numpy()[:, labels_data_frame.columns.to_list().index('Codes')]

        # Create classifier instances for the frontal image and the tiles
        classifier = [None] * (self.total_tiles_no + 1)
        for tile_name, tile_no in [('original', 0)] + [('tile_' + str(tile_no), tile_no) for tile_no in range(1, 1 + self.total_tiles_no)]:
            classifier[tile_no] = classification.Classifier(load_dir='./model/' + tile_name, configurations=self.config)

        # Repeat for all test benchmarks
        for test in range(benchmarks_no * self.variations):

            # Print the current benchmark number
            print('Test#: {}'.format(test))

            # Set test directory name
            test_dir = str(int(test / self.variations))

            # Read label and encode it
            with open(os.path.join('./test_set/', test_dir, "label.txt"), "r") as label_file:
                label = label_file.read()
            label_code = label_codes[np.where(label == label_names)[0][0]]

            # Set the frontal image filename and read it
            if (test % self.variations) >= int(self.variations / 2):
                frontal_image_address = os.path.join('./test_set/', test_dir, 'Front_Down.jpg')
            else:
                frontal_image_address = os.path.join('./test_set/', test_dir, 'Front_Up.jpg')
            frontal_image = cv.imread(frontal_image_address)

            # Randomly select and read an overlay image
            random_benchmark = str(np.random.randint(low=0, high=benchmarks_no))
            if (test % self.variations) >= int(self.variations / 2):
                random_image_address = os.path.join('./test_set/', random_benchmark, 'Front_Down.jpg')
            else:
                random_image_address = os.path.join('./test_set/', random_benchmark, 'Front_Up.jpg')
            random_image = cv.imread(random_image_address)

            # Read the frontal and random depth maps
            frontal_depth_map, random_depth_map = surface_normal.get_depth_map(test_number=test, random_test=random_benchmark)

            # Modify the frontal image via data augmentation techniques, if needed
            frontal_image = test_augmentation.manipulate_images(input_image=frontal_image,
                                                                test_number=test,
                                                                overlay_image=random_image,
                                                                whiteout_blackout_ratio=self.config["whiteout_blackout_ratio"],
                                                                superimpose_ratio=self.config["superimpose_ratio"],
                                                                brightness_change=self.config["brightness_change"],
                                                                noise_std_mean_lighter=self.config["noise_std_mean_lighter"],
                                                                noise_std_mean_heavier=self.config["noise_std_mean_heavier"],
                                                                blur_kernel_lighter=self.config["blur_kernel_lighter"],
                                                                blur_kernel_heavier=self.config["blur_kernel_heavier"])

            # Modify the frontal depth map via augmentation techniques, corresponding to those for color frontal image
            frontal_depth_map = test_augmentation.manipulate_depth_maps(depth_map=frontal_depth_map,
                                                                        test_number=test,
                                                                        overlay_depth_map=random_depth_map,
                                                                        average_out_ratio=self.config["whiteout_blackout_ratio"],
                                                                        superimpose_ratio=self.config["superimpose_ratio"])

            # Classify the frontal image
            frontal_probabilities = classifier[0].classify(frontal_image)
            if self.config["decision_fusion_type"] == 'Dempster-Shafer':
                frontal_probabilities_unscaled, frontal_universal = classifier[0].get_dst_masses()

            # Compute entropy of the frontal image
            # gray_frontal = cv.cvtColor(frontal_image, cv.COLOR_BGR2GRAY)
            # frontal_hist = np.squeeze(cv.calcHist(gray_frontal, channels=(0,), histSize=(128,), ranges=(0, 256), mask=None))
            # frontal_hist /= np.sum(frontal_hist)
            # frontal_entropy = - np.sum(frontal_hist * np.log2(frontal_hist + np.finfo(float).eps))

            # For each tile
            for tile_no in range(self.total_tiles_no):

                # Extract the tile image and depth map
                tile_image = extract_tile(frontal_image, tile_no + 1)
                tile_depth_map = extract_tile(frontal_depth_map, tile_no + 1)

                # Classify the tile
                tile_probabilities_specific[tile_no] = classifier[tile_no].classify(tile_image)
                tile_probabilities_general[tile_no] = classifier[0].classify(tile_image)

                # Compute average depth component of surface normal
                surface_normal_score[tile_no] = surface_normal.compute_foreshortening_score(tile_depth_map)

                # Calculate histogram of grayscale tile
                gray_tile = cv.cvtColor(tile_image, cv.COLOR_BGR2GRAY)
                tile_hist = np.squeeze(cv.calcHist([gray_tile], channels=(0,), histSize=(128,), ranges=(0, 256), mask=None))
                tile_hist /= np.sum(tile_hist)

                # Compute histogram variance of the tile
                hist_mean = np.sum(tile_hist * np.arange(0, 256, 2))
                hist_variance[tile_no] = np.sum(((np.arange(0, 256, 2) - hist_mean) ** 2) * tile_hist)

                # Compute histogram uniformity of the tile (additive inverse of gini index)
                hist_uniformity[tile_no] = - (1 - np.sum(tile_hist ** 2))

                # Compute histogram negative entropy of the tile
                hist_negative_entropy[tile_no] = np.sum(tile_hist * np.log2(tile_hist + np.finfo(float).eps))

                # Read the side image, corresponding to the tile
                side_image_address = os.path.join('./test_set/', test_dir, str(tile_no + 1) + '.jpg')
                side_image = cv.imread(side_image_address)

                # Classify the side image
                side_probabilities[tile_no] = classifier[0].classify(side_image)
                if self.config["decision_fusion_type"] == 'Dempster-Shafer':
                    side_probabilities_unscaled, side_universal = classifier[0].get_dst_masses()

                # Fuse the classification results of the frontal and side images
                if self.config["decision_fusion_type"] == 'Dempster-Shafer':
                    fused_probabilities[tile_no] = fusion.fuse(input_1=frontal_probabilities_unscaled,
                                                               input_2=side_probabilities_unscaled,
                                                               fusion_type=self.config["decision_fusion_type"],
                                                               universal_mass_1=frontal_universal,
                                                               universal_mass_2=side_universal)
                else:
                    fused_probabilities[tile_no] = fusion.fuse(input_1=frontal_probabilities,
                                                               input_2=side_probabilities[tile_no],
                                                               fusion_type=self.config["decision_fusion_type"])

            # Save the results
            csv_data[test] = np.array([test,
                                       frontal_probabilities,
                                       side_probabilities.copy(),
                                       fused_probabilities.copy(),
                                       tile_probabilities_specific.copy(),
                                       tile_probabilities_general.copy(),
                                       hist_uniformity.copy(),
                                       hist_variance.copy(),
                                       hist_negative_entropy.copy(),
                                       surface_normal_score.copy(),
                                       label_code], dtype=object)

        # Save the scores and probabilities in a numpy file
        np.save('./results/scores.npy', csv_data)

        # Save the scores and probabilities in a CSV file
        df = pd.DataFrame(data=csv_data, columns=['Test Number',
                                                  'Front Probability',
                                                  'Side Probabilities',
                                                  'Fused Probabilities',
                                                  'Tile Probabilities (Dedicated Classifier)',
                                                  'Tile Probabilities (Common Classifier)',
                                                  'Histogram Uniformity',
                                                  'Histogram Variance',
                                                  'Histogram Negative Entropy',
                                                  'Surface Normal Score',
                                                  'Label'])
        df.to_csv('./results/scores.csv', index=False)

    def train_vision(self):
        """
        Train vision model.
        """

        vision_training.train_vision(self.config)

    def evaluate_next_best_view(self):
        """
        Evaluate the next best view mechanism.
        """

        evaluation.evaluate(self.config, compute_next_best_view, classification_sum_absolute_difference,
                            classification_negative_entropy, classification_kl_divergence, self.total_tiles_no)

    def read_config(self, config_file, config_section='DEFAULT'):
        """
        Read configurations from the config file.
        """

        # Load config parser
        cfg_parser = configparser.ConfigParser()

        # Read the config file
        list_of_files_read = cfg_parser.read(os.path.join('./config/', config_file))

        # Make sure at least a configuration file was read
        if len(list_of_files_read) <= 0:
            raise Exception("Fatal Error: No configuration file " + config_file + " found in ./config/.")

        # Load the configurations
        self.config["whiteout_blackout_ratio"] = cfg_parser.getfloat(config_section, "whiteout_blackout_ratio")
        self.config["superimpose_ratio"] = cfg_parser.getfloat(config_section, "superimpose_ratio")
        self.config["brightness_change"] = cfg_parser.getint(config_section, "brightness_change")
        self.config["noise_std_mean_lighter"] = cfg_parser.getint(config_section, "noise_std_mean_lighter")
        self.config["noise_std_mean_heavier"] = cfg_parser.getint(config_section, "noise_std_mean_heavier")
        self.config["blur_kernel_lighter"] = cfg_parser.getint(config_section, "blur_kernel_lighter")
        self.config["blur_kernel_heavier"] = cfg_parser.getint(config_section, "blur_kernel_heavier")
        self.config["image_fix_size"] = ast.literal_eval(cfg_parser.get(config_section, "image_fix_size"))

        self.config["confidence_ratio"] = cfg_parser.getfloat(config_section, "confidence_ratio")
        self.config["recognition_threshold"] = cfg_parser.getfloat(config_section, "recognition_threshold")
        self.config["confidence_ratio_sweep_max"] = cfg_parser.getfloat(config_section, "confidence_ratio_sweep_max")
        self.config["confidence_ratio_sweep_no"] = cfg_parser.getint(config_section, "confidence_ratio_sweep_no")
        self.config["recognition_threshold_sweep_no"] = cfg_parser.getint(config_section, "recognition_threshold_sweep_no")


        self.config["classifier_type"] = cfg_parser.get(config_section, "classifier_type")
        self.config["svm_kernel"] = cfg_parser.get(config_section, "svm_kernel")
        self.config["rf_criterion"] = cfg_parser.get(config_section, "rf_criterion")
        self.config["rf_estimators_no"] = cfg_parser.getint(config_section, "rf_estimators_no")
        self.config["nn_network_architecture"] = cfg_parser.get(config_section, "nn_network_architecture")
        self.config["nn_epochs"] = cfg_parser.getint(config_section, "nn_epochs")
        self.config["nn_max_learning_rate"] = cfg_parser.getfloat(config_section, "nn_max_learning_rate")
        self.config["nn_batch_size"] = cfg_parser.getint(config_section, "nn_batch_size")
        self.config["nn_validation_split"] = cfg_parser.getfloat(config_section, "nn_validation_split")
        self.config["nn_early_stopping_patience"] = cfg_parser.getint(config_section, "nn_early_stopping_patience")
        self.config["svm_feature_types"] = ast.literal_eval(cfg_parser.get(config_section, "svm_feature_types"))
        self.config["hog_reduced_features_no"] = cfg_parser.getint(config_section, "hog_reduced_features_no")
        self.config["hog_window_size"] = ast.literal_eval(cfg_parser.get(config_section, "hog_window_size"))
        self.config["hog_block_size"] = ast.literal_eval(cfg_parser.get(config_section, "hog_block_size"))
        self.config["hog_block_stride"] = ast.literal_eval(cfg_parser.get(config_section, "hog_block_stride"))
        self.config["hog_cell_size"] = ast.literal_eval(cfg_parser.get(config_section, "hog_cell_size"))
        self.config["hog_bin_no"] = cfg_parser.getint(config_section, "hog_bin_no")
        self.config["color_histogram_size"] = cfg_parser.getint(config_section, "color_histogram_size")
        self.config["bag_of_words_feature_type"] = cfg_parser.get(config_section, "bag_of_words_feature_type")
        self.config["bag_of_words_cluster_no"] = cfg_parser.getint(config_section, "bag_of_words_cluster_no")
        self.config["sift_features_no"] = cfg_parser.getint(config_section, "sift_features_no")
        self.config["sift_octave_layers"] = cfg_parser.getint(config_section, "sift_octave_layers")
        self.config["sift_contrast_threshold"] = cfg_parser.getfloat(config_section, "sift_contrast_threshold")
        self.config["sift_edge_threshold"] = cfg_parser.getfloat(config_section, "sift_edge_threshold")
        self.config["sift_sigma"] = cfg_parser.getfloat(config_section, "sift_sigma")
        self.config["kaze_threshold"] = cfg_parser.getfloat(config_section, "kaze_threshold")
        self.config["kaze_octaves_no"] = cfg_parser.getint(config_section, "kaze_octaves_no")
        self.config["kaze_octave_layers"] = cfg_parser.getint(config_section, "kaze_octave_layers")
        self.config["decision_fusion_type"] = cfg_parser.get(config_section, "decision_fusion_type")
        self.config["dst_universal_class_ratio_to_dataset"] = cfg_parser.getfloat(config_section, "dst_universal_class_ratio_to_dataset")
        self.config["dst_augment_universal_class"] = cfg_parser.getboolean(config_section, "dst_augment_universal_class")
        self.config["train_images_extension"] = cfg_parser.get(config_section, "train_images_extension")
        self.config["cross_validation_splits"] = cfg_parser.getint(config_section, "cross_validation_splits")

        # Check if at least one of HOG, color histogram, and Hu moments is enabled if the classifier is SVM
        if self.config['classifier_type'] == 'SVM' and len(self.config["svm_feature_types"]) == 0:
            raise Exception("For SVM classifier, at least one feature type should be specified.")


def extract_tile(image, tile_no):
    """
    Crop a tile from the input image.
    :param image: Input image (3D or 2D array; for 3D: row, column, channel; for 2D: row, column)
    :param tile_no: Tile number (int)
    :return: Extracted tile (3D or 2D; for 3D: row, column, channel; for 2D: row, column)
    """

    # Get number of rows and columns
    rows = image.shape[0]
    columns = image.shape[1]

    # Choose the cropping operation
    if tile_no == 1:
        tile = image[: int(rows / 3), : int(columns / 3)]
    elif tile_no == 2:
        tile = image[: int(rows / 3), int(columns / 3): int(2 * columns / 3)]
    elif tile_no == 3:
        tile = image[: int(rows / 3), int(2 * columns / 3):]
    elif tile_no == 4:
        tile = image[int(rows / 3): int(2 * rows / 3), : int(columns / 3)]
    elif tile_no == 5:
        tile = image[int(rows / 3): int(2 * rows / 3), int(2 * columns / 3):]
    elif tile_no == 6:
        tile = image[int(2 * rows / 3):, : int(columns / 3)]
    elif tile_no == 7:
        tile = image[int(2 * rows / 3):, int(columns / 3): int(2 * columns / 3)]
    elif tile_no == 8:
        tile = image[int(2 * rows / 3):, int(2 * columns / 3):]
    else:
        raise Exception('Tile number ' + str(tile_no) + ' is not expected.')

    return tile


def classification_sum_absolute_difference(tile_probs, front_probs):
    """
    Measure the sum of absolute difference (SAD) of classifications of the frontal view and a tile.
    :param tile_probs: Classification probabilities of the tile (1D array)
    :param front_probs: Classification probabilities of the frontal view (1D array)
    :return: Classification SAD (float)
    """

    sad = np.sum(np.abs(np.array(tile_probs) - np.array(front_probs)))

    return sad


def classification_negative_entropy(tile_probs):
    """
    Measure the negative entropy of classifications of a tile.
    :param tile_probs: Classification probabilities of the tile (1D array)
    :return: Classification information gain (float)
    """

    tile_probs = np.array(tile_probs)
    negative_entropy = np.sum(tile_probs * np.log2(tile_probs + np.finfo(float).eps))

    return negative_entropy


def classification_kl_divergence(tile_probs, front_probs):
    """
    Measure the KL divergence of classifications of the frontal view and a tile.
    :param tile_probs: Classification probabilities of the tile (1D array)
    :param front_probs: Classification probabilities of the frontal view (1D array)
    :return: Classification KL divergence (float)
    """

    front_probs, tile_probs = np.array(front_probs), np.array(tile_probs)
    kl_divergence = np.sum(front_probs * np.log2(front_probs + np.finfo(float).eps)) - np.sum(front_probs * np.log2(tile_probs + np.finfo(float).eps))

    return kl_divergence


def compute_next_best_view(criteria):
    """
    Choose the tile with the highest votes among the criteria.
    :param criteria: Criteria in deciding the next best view. It is an array of array of floats, with each element of
    the outer array being a criterion, and each element of that being a float value that are the score of tiles.
    :return: Winner tile number, Histogram of votes to tiles
    """

    # Create a histogram of votes
    combined_histogram, _ = np.histogram(np.concatenate([[x] * i for measure in criteria for i, x in enumerate(np.argsort(measure))]), bins=np.arange(0, 6))

    # Add noise, less than 1, to the vote counts to resolve ties in a random manner
    combined_measure_scores = combined_histogram + np.random.rand(combined_histogram.size)

    # Find the winner tile
    winner_tile = np.argmax(combined_measure_scores)

    return winner_tile, combined_measure_scores
