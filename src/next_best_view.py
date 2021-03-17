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


class NextBestView:

    def __init__(self):
        """
        Constructor
        """

        # Initialize variables
        self.config = {}

        # Read configurations
        self.read_config('./config/config.cfg')

    def generate_scores(self):
        """
        Generate class probabilities of frontal and side images, and generate the scores of each tile in the frontal image.
        """

        # Set number of benchmarks, test variations per benchmark, and tiles
        benchmarks_no = len([directory for directory in os.listdir('./test_set/') if os.path.isdir(directory)])
        variations = 30
        total_tiles_no = 5

        # Initialize variables
        tile_image = [] * total_tiles_no
        tile_depth_map = [] * total_tiles_no
        tile_probabilities_specific = [] * total_tiles_no
        tile_probabilities_general = [] * total_tiles_no
        side_probabilities = [] * total_tiles_no
        fused_probabilities = [] * total_tiles_no
        surface_normal_score = [] * total_tiles_no
        converted_hist_variance = [] * total_tiles_no
        converted_hist_moment_3 = [] * total_tiles_no
        hist_uniformity = [] * total_tiles_no
        hist_entropy = [] * total_tiles_no
        csv_data = np.zeros((benchmarks_no * variations, 11), dtype=np.float)

        # Create classifier instances for the frontal image and the tiles
        classifier = [] * 6
        for tile_name, tile_no in [('original', 0)] + [('tile_' + str(tile_no), tile_no) for tile_no in range(1, 1 + total_tiles_no)]:
            classifier[tile_no] = classification.Classifier(load_dir='./model/' + tile_name, configurations=self.config)

        # Repeat for all test benchmarks
        for test in range(benchmarks_no * variations):

            # Print the current benchmark number
            print('\nTest#: {}\n'.format(test))

            # Set test directory name
            test_dir = str(int(test / variations))

            # Set the frontal image filename and read it
            if (test % variations) >= int(variations / 2):
                frontal_image_address = os.path.join('./test_set/', test_dir, 'Front_Down.jpg')
            else:
                frontal_image_address = os.path.join('./test_set/', test_dir, 'Front_Up.jpg')
            frontal_image = cv.imread(frontal_image_address)

            # Randomly select and read an overlay image
            random_benchmark = str(np.random.randint(low=0, high=benchmarks_no))
            if (test % variations) >= int(variations / 2):
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

            # For each tile
            for tile_no in range(1, 1 + total_tiles_no):

                # Extract the tile image and depth map
                tile_image[tile_no] = extract_tile(frontal_image, tile_no)
                tile_depth_map[tile_no] = extract_tile(frontal_depth_map, tile_no)

                # Classify the tile
                tile_probabilities_specific[tile_no] = classifier[tile_no].classify(tile_image[tile_no])
                tile_probabilities_general[tile_no] = classifier[0].classify(tile_image[tile_no])

                # Compute average depth component of surface normal
                surface_normal_score[tile_no] = surface_normal.compute_average_surface_normal_depth(tile_depth_map[tile_no])

                # Calculate histogram of grayscale tile
                gray_tile = cv.cvtColor(tile_image[tile_no], cv.COLOR_BGR2GRAY)
                tile_hist = np.squeeze(cv.calcHist(gray_tile, channels=(0,), histSize=(128,), ranges=(0, 256), mask=None))
                tile_hist /= np.sum(tile_hist)

                # Compute histogram variance of the tile
                hist_mean = np.sum(tile_hist * np.arange(0, 256, 2))
                hist_variance = np.sum(((np.arange(0, 256, 2) - hist_mean) ** 2) * tile_hist)
                converted_hist_variance[tile_no] = 1 - (1.0 / (1 + hist_variance))

                # Compute histogram third moment of the tile
                hist_moment_3 = np.sum(((np.arange(0, 256, 2) - hist_mean) ** 3) * tile_hist)
                converted_hist_moment_3[tile_no] = 1.0 / (1 + np.abs(hist_moment_3))

                # Compute uniformity score of the tile
                hist_uniformity[tile_no] = np.sum(tile_hist ** 2)

                # Compute histogram entropy of the tile
                hist_entropy[tile_no] = - np.sum(tile_hist * np.log2(tile_hist))

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
                                       side_probabilities,
                                       fused_probabilities,
                                       tile_probabilities_specific,
                                       tile_probabilities_general,
                                       hist_uniformity,
                                       converted_hist_variance,
                                       converted_hist_moment_3,
                                       hist_entropy,
                                       surface_normal_score])

        # Save the scores and probabilities in a CSV file
        df = pd.DataFrame(data=csv_data, columns=['Test Number',
                                                  'Front Probability',
                                                  'Side Probabilities',
                                                  'Fused Probabilities',
                                                  'Tile Probabilities (Specific)',
                                                  'Tile Probabilities (General)',
                                                  'Uniformity',
                                                  'STD (R)',
                                                  'Converted Moment 3',
                                                  'Entropy',
                                                  'Surface Normal Score'], dtype=np.float)
        df.to_csv('./results/scores.csv')

    def train_vision(self):
        """
        Train vision model.
        """

        vision_training.train_vision(self.config)

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
        self.config["classifier_type"] = cfg_parser.get(config_section, "classifier_type")
        self.config["svm_kernel"] = cfg_parser.get(config_section, "svm_kernel")
        self.config["rf_criterion"] = cfg_parser.get(config_section, "rf_criterion")
        self.config["rf_estimators_no"] = cfg_parser.getint(config_section, "rf_estimators_no")
        self.config["nn_network_architecture"] = cfg_parser.getint(config_section, "nn_network_architecture")
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
        self.config["train_vision_first"] = cfg_parser.getboolean(config_section, "train_vision_first")
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
