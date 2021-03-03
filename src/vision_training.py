import numpy as np
import cv2 as cv
import os


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

        # Read images in the current folder
        image_no = -1
        for image_no, image in enumerate(current_image_list):
            dataset_labels[sample_no + image_no] = class_no
            dataset_images[sample_no + image_no] = cv.imread(os.path.join(current_address, image), cv.IMREAD_COLOR)

        # Update the number of samples written so far
        sample_no += image_no + 1

    # Remove unused elements in the dataset arrays
    dataset_labels = dataset_labels[: sample_no]
    dataset_images = dataset_images[: sample_no]

    # Print number of read samples
    print(str(sample_no) + ' training images read.')

    return dataset_images, dataset_labels, label_code
