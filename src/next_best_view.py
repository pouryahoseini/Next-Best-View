import numpy as np
import configparser
import os
import pandas as pd
import src.vision_training as vision_training
import src.train_data_augmentation as train_augmentation


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
        :return: trained model
        """

        # Set the vision training root address
        root_address = './vision_training/'

        # Read the dataset from files on drive
        dataset_images, dataset_labels, label_codes = vision_training.get_train_samples(root_address=root_address,
                                                                                       image_size=self.config["train_image_fix_size"],
                                                                                       train_images_extension=self.config["train_images_extension"])

        # Data augmentation
        augmented_dataset_images, augmented_dataset_labels = train_augmentation.train_data_augmentation(dataset_images, dataset_labels)

        # In the case of Dempster-Shafer fusion, add the 'Universal' class to the dataset
        if self.config["decision_fusion_type"] == 'DST':
            dataset_images, dataset_labels, label_codes = vision_training.create_dempster_shafer_dataset(dataset_images, dataset_labels, label_codes,
                                                                                                         augmented_dataset_images=augmented_dataset_images,
                                                                                                         augmented_dataset_labels=augmented_dataset_labels,
                                                                                                         universal_class_ratio_to_dataset=self.config["dst_universal_class_ratio_to_dataset"],
                                                                                                         dst_augment_universal_class=self.config["dst_augment_universal_class"])
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
        # if self.config["classifier_type"] == 'SVM':
        #
        # elif self.config["classifier_type"] == 'RF':
        #
        # elif self.config["classifier_type"] == 'NN':
        #
        # else:
        #     raise Exception("Classifier type ' + classifier_type + ' not recognized.")

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
        self.config["bag_of_words_descriptor"] = cfg_parser.get(config_section, "bag_of_words_descriptor")
        self.config["hog_reduced_features_no"] = cfg_parser.getint(config_section, "hog_reduced_features_no")
        self.config["decision_fusion_type"] = cfg_parser.get(config_section, "decision_fusion_type")
        self.config["dst_universal_class_ratio_to_dataset"] = cfg_parser.getfloat(config_section, "dst_universal_class_ratio_to_dataset")
        self.config["dst_augment_universal_class"] = cfg_parser.getboolean(config_section, "dst_augment_universal_class")
        self.config["train_vision_first"] = cfg_parser.getboolean(config_section, "train_vision_first")
        self.config["train_image_fix_size"] = cfg_parser.getint(config_section, "train_image_fix_size")
        self.config["train_images_extension"] = cfg_parser.get(config_section, "train_images_extension")
        self.config["tiles_measurement_file"] = cfg_parser.get(config_section, "tiles_measurement_file")

