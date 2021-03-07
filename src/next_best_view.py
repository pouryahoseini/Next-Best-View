import configparser
import os
import ast
import src.vision_training as vision_training


class NextBestView:

    def __init__(self):
        """
        Constructor
        """

        # Initialize variables
        self.config = {}

        # Read configurations
        self.read_config('./config/config.cfg')

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
        self.config["classifier_type"] = cfg_parser.get(config_section, "classifier_type")
        self.config["svm_kernel"] = cfg_parser.get(config_section, "svm_kernel")
        self.config["rf_criterion"] = cfg_parser.get(config_section, "rf_criterion")
        self.config["rf_estimators_no"] = cfg_parser.getint(config_section, "rf_estimators_no")
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
        self.config["train_image_fix_size"] = ast.literal_eval(cfg_parser.get(config_section, "train_image_fix_size"))
        self.config["train_images_extension"] = cfg_parser.get(config_section, "train_images_extension")
        self.config["cross_validation_splits"] = cfg_parser.getint(config_section, "cross_validation_splits")
        self.config["tiles_measurement_file"] = cfg_parser.get(config_section, "tiles_measurement_file")

        # Check if at least one of HOG, color histogram, and Hu moments is enabled if the classifier is SVM
        if self.config['classifier_type'] == 'SVM' and len(self.config["svm_feature_types"]) == 0:
            raise Exception("For SVM classifier, at least one feature type should be specified.")
