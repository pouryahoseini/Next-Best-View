import numpy as np
import cv2 as cv


def train_data_augmentation(input_dataset):
    """
    Creates a dataset with augmented samples.
    :param input_dataset: Input dataset (3D numpy array (image_no, rows, columns))
    :return: Augmented dataset
    """

    # Initialize the augmented dataset
    number_of_augmentations = 22   # 5 + 3 + 6 + 2 + 5 + 1 (the last one is the original sample)
    augmented_dataset = np.zeros((number_of_augmentations * input_dataset.shape[0], input_dataset.shape[1], input_dataset.shape[2]), dtype=input_dataset.dtype)

    # Repeat for all the samples in the dataset
    for sample_no, sample in enumerate(input_dataset):

        # Set the current index of the augmented dataset
        current_index = sample_no * number_of_augmentations

        # Add the original image
        augmented_dataset[current_index] = input_dataset[sample_no]
        current_index += 1

        # Crop image
        for crop_counter in range(5):
            augmented_dataset[current_index + crop_counter] = crop_sample(input_dataset[sample_no], crop_counter)
        current_index += crop_counter + 1

        # Blur image
        for blurring_counter in range(3):
            augmented_dataset[current_index + blurring_counter] = blur_sample(input_dataset[sample_no], blurring_counter)
        current_index += blurring_counter + 1

        # Change brightness
        for brightness_counter in range(6):
            augmented_dataset[current_index + brightness_counter] = change_brightness(input_dataset[sample_no], brightness_counter)
        current_index += brightness_counter + 1

        # Add noise
        for noise_counter in range(2):
            augmented_dataset[current_index + noise_counter] = add_noise(input_dataset[sample_no], noise_counter + 1, noise_std = 0.1)
        current_index += noise_counter + 1

        # Geometric transform
        for geometric_transform_counter in range(5):
            augmented_dataset[current_index + geometric_transform_counter] = transform_geometrically(input_dataset[sample_no], geometric_transform_counter)

    return augmented_dataset


def crop_sample(input_image, crop_mode):
    """
    Crop the input images.
    :param input_image: Input image (2D numpy array)
    :param crop_mode: Defines how to crop (0: zoom by a factor of 2, 1: crop one-third from the top, 2: crop one-third from the bottom, 3: crop one-third from left, 4: crop one-third from right)
    :return: Cropped image
    """

    # Save the input image dimensions
    width = input_image.shape[1]
    height = input_image.shape[0]

    # Keep the crop mode in the legal range
    crop_mode = crop_mode % 5

    # Determine the crop operation
    if crop_mode == 0:  # zoom by a factor of 2
        processed_image = cv.resize(input_image[height / 4: (3 * height) / 4, width / 4: (3 * width) / 4], input_image.shape)
    elif crop_mode == 1:    # crop one-third from the top
        processed_image = cv.resize(input_image[height / 3:, :], input_image.shape)
    elif crop_mode == 2:    # crop one-third from the bottom
        processed_image = cv.resize(input_image[0: (2 * height) / 3, :], input_image.shape)
    elif crop_mode == 3:    # crop one-third from left
        processed_image = cv.resize(input_image[:, width / 3:], input_image.shape)
    else:    # crop one-third from right
        processed_image = cv.resize(input_image[:, : (2 * width) / 3], input_image.shape)

    return processed_image


def blur_sample(input_image, blur_mode):
    """
    Blur the input images.
    :param input_image: Input image (2D numpy array)
    :param blur_mode: Defines how to blur (0: omni-directional, 1: horizontal blurring, 2: vertical blurring)
    :return: Blurred image
    """

    # Keep the blur mode in the legal range
    blur_mode = blur_mode % 3

    # Determine the blur operation
    if blur_mode == 0:  # omni-directional blurring
        processed_image = cv.GaussianBlur(input_image, (7, 7), 2)
    elif blur_mode == 1:    # horizontal blurring
        processed_image = cv.GaussianBlur(input_image, (7, 1), 2)
    else:   # vertical blurring
        processed_image = cv.GaussianBlur(input_image, (1, 7), 2)

    return processed_image


def change_brightness(input_image, brightness_mode):
    """
    Change brightness of an image.
    :param input_image: Input image (2D numpy array)
    :param brightness_mode: Defines how to change brightness (0: +40, 1: +80, 2: +120, 3: -40, 4: -80, 5: -120)
    :return: Image with modified brightness
    """

    # Keep the brightness mode in the legal range
    brightness_mode = brightness_mode % 6

    # Convert the input image to the HSV color domain
    hsv_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)

    # Determine the brightness operation
    if brightness_mode == 0:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] + 40
    elif brightness_mode == 1:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] + 80
    elif brightness_mode == 2:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] + 120
    elif brightness_mode == 3:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] - 40
    elif brightness_mode == 4:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] - 80
    else:
        hsv_image[:, :, 2] = hsv_image[:, :, 2] - 120

    processed_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return processed_image


def add_noise(input_image, hsv_channel, noise_std = 0.1):
    """
    Add noise to an image.
    :param input_image: Input image (2D numpy array)
    :param hsv_channel: Noise is added to this channel of the HSV color space
    :param noise_std: Standard deviation of the noise
    :return: Image with added noise
    """

    # Check if the selected HSV channel is correct
    assert ((hsv_channel >= 0) and (hsv_channel <= 2)), 'HSV channel to add noise to is out of range.'

    # Extract the HSV channels
    converted_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)

    # Convert the channel image to float and between 0 and 1
    channel = converted_image[:, :, hsv_channel].astype(np.float) / 255.0

    # Preserving the original image's dynamic range
    min_pixel_value, max_pixel_value, _, _ = cv.minMaxLoc(channel)

    # Add Gaussian noise
    noise = np.zeros(channel.shape, dtype=channel.dtype)
    cv.randn(noise, 0, noise_std)
    channel = channel + noise

    # Normalize the image to ensure the pixel values would not go more than the maximum possible value
    channel = cv.normalize(channel, None, min_pixel_value, max_pixel_value, cv.NORM_MINMAX)

    # Convert the image back to uint type
    channel = (channel * 255.0).astype(converted_image.dtype)

    # Merge channels
    converted_image[:, :, hsv_channel] = channel

    # Convert the image to BGR domain
    processed_image = cv.cvtColor(converted_image, cv.COLOR_HSV2BGR)

    return processed_image


def transform_geometrically(input_image, transform_type):
    """
    Transform an image geometrically (i.e. flip and rotation)
    :param input_image: Input image (2D numpy array)
    :param transform_type: Defines how to transform the image (0: 90 degrees rotation, 1: 180 degrees rotation, 2: 270 degrees rotation, 3: vertical flip, 4: horizontal flip)
    :return: Geometrically transformed image
    """

    # Keep the transform type in the legal range
    transform_type %= 5

    # Determine the geometric transformation
    if transform_type == 0:
        processed_image = cv.rotate(input_image, cv.ROTATE_90_CLOCKWISE)
    elif transform_type == 1:
        processed_image = cv.rotate(input_image, cv.ROTATE_180)
    elif transform_type == 2:
        processed_image = cv.rotate(input_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif transform_type == 3:
        processed_image = cv.flip(input_image, 0)
    else:
        processed_image = cv.flip(input_image, 1)
    return processed_image
