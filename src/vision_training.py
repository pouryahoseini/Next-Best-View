import numpy as np
import cv2 as cv
import os


def train_vision():
    pass


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
