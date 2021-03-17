import numpy as np
import cv2 as cv


def manipulate_images(input_image, test_number, overlay_image, whiteout_blackout_ratio, superimpose_ratio, brightness_change, noise_std_mean_lighter, noise_std_mean_heavier, blur_kernel_lighter, blur_kernel_heavier):
    """
    Manipulate images to create new ones that are harder to classify to mimic difficult object recognition conditions.
    :param input_image: Input image to be manipulated
    :param test_number: The current test number that specifies which type of image manipulation should be done
    :param overlay_image: The image to be superimposed, in the case of a relevant image manipulation type
    :param whiteout_blackout_ratio: Ratio of the x and y axes in the input image to be covered with white/black areas
    :param superimpose_ratio: Ratio of the x and y axes in the input image to be superimposed with another image
    :param brightness_change: Brightness change
    :param noise_std_mean_lighter: Mean and standard deviation of noise for lighter noise additions
    :param noise_std_mean_heavier: Mean and standard deviation of noise for heavier noise additions
    :param blur_kernel_lighter: Blurring kernel size for lighter blurring
    :param blur_kernel_heavier: Blurring kernel size for heavier blurring
    :return: Manipulated image
    """

    # Identify what type of manipulation to perform
    manipulation_type_no = 15
    variation = test_number % int(manipulation_type_no)

    # Copy the input image on the output image
    new_image = np.copy(input_image)

    # Initialize variables
    noise = np.zeros(input_image.shape, dtype=input_image.dtype)

    # Decide on the size of the superimposed region
    if int(input_image.shape[1] * superimpose_ratio) > overlay_image.shape[1]:
        overlay_column = overlay_image.shape[1]
    else:
        overlay_column = int(input_image.shape[1] * superimpose_ratio)

    if int(input_image.shape[0] * superimpose_ratio) > overlay_image.shape[0]:
        overlay_row = overlay_image.shape[0]
    else:
        overlay_row = int(input_image.shape[0] * superimpose_ratio)

    # Select the appropriate image manipulation
    # Note: variation == 0 is reserved for the original image
    if variation == 0:
        pass
    elif variation == 1:    # Top whiteout
        cv.rectangle(new_image, (0, 0), (input_image.shape[1], int(input_image.shape[0] * whiteout_blackout_ratio)), (255, 255, 255), -1)
    elif variation == 2:   # Bottom whiteout
        cv.rectangle(new_image, (0, int(input_image.shape[0] * (1 - whiteout_blackout_ratio))), (input_image.shape[1], input_image.shape[0]), (255, 255, 255), -1)
    elif variation == 3:  # Left blackout
        cv.rectangle(new_image, (0, 0), (int(input_image.shape[1] * whiteout_blackout_ratio), input_image.shape[0]), (0, 0, 0), -1)
    elif variation == 4:   # Right blackout
        cv.rectangle(new_image, (int(input_image.shape[1] * (1 - whiteout_blackout_ratio)), 0), (input_image.shape[1], input_image.shape[0]), (0, 0, 0), -1)
    elif variation == 5:   # Add noise, weaker
        cv.randn(noise, (noise_std_mean_lighter, noise_std_mean_lighter, noise_std_mean_lighter), (noise_std_mean_lighter, noise_std_mean_lighter, noise_std_mean_lighter))
        new_image += noise
    elif variation == 6:    # Add noise, stronger
        cv.randn(noise, (noise_std_mean_heavier, noise_std_mean_heavier, noise_std_mean_heavier), (noise_std_mean_heavier, noise_std_mean_heavier, noise_std_mean_heavier))
        new_image += noise
    elif variation == 7:    # Superimpose top left with bottom right of overlay image
        new_image[: overlay_row, : overlay_column] = overlay_image[- overlay_row:, - overlay_column:]
    elif variation == 8:   # Superimpose top right with bottom left of overlay image
        new_image[: overlay_row, - overlay_column:] = overlay_image[- overlay_row:, : overlay_column]
    elif variation == 9:    # Superimpose bottom left with top right of overlay image
        new_image[- overlay_row:, : overlay_column] = overlay_image[: overlay_row, - overlay_column:]
    elif variation == 10:   # Superimpose bottom right with top left of overlay image
        new_image[- overlay_row:, - overlay_column:] = overlay_image[: overlay_row, : overlay_column]
    elif variation == 11:   # Blur, weaker
        new_image = cv.medianBlur(input_image, blur_kernel_lighter)
    elif variation == 12:   # Blur, stronger
        new_image = cv.medianBlur(input_image, blur_kernel_heavier)
    elif variation == 13:   # Illumination, bright
        new_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
        channels = cv.split(new_image)
        channels[2] = channels[2] + brightness_change
        new_image = cv.merge(channels)
        new_image = cv.cvtColor(new_image, cv.COLOR_HSV2BGR)
    elif variation == 14:   # Illumination, dark
        new_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
        channels = cv.split(new_image)
        channels[2] = channels[2] - brightness_change
        new_image = cv.merge(channels)
        new_image = cv.cvtColor(new_image, cv.COLOR_HSV2BGR)
    else:
        raise Exception("Variation type not recognized.")

    return new_image


def manipulate_depth_maps(depth_map, test_number, overlay_depth_map, average_out_ratio, superimpose_ratio):
    """
    Modify depth maps based on the alterations in the corresponding color image.
    :param depth_map: Input depth map to be manipulated
    :param test_number: The current test number that specifies which type of depth map manipulation should be done
    :param overlay_depth_map: The depth map to be superimposed, in the case of a relevant depth map manipulation type
    :param average_out_ratio: Ratio of the x and y axes in the input depth map to be covered with the average depth of the image
    :param superimpose_ratio: Ratio of the x and y axes in the input depth map to be superimposed with another depth map
    :return: Manipulated depth map
    """

    # Identify the modification type
    variation = test_number % 15

    # Compute the average depth
    mean_depth = np.mean(depth_map)

    # Determine dimensions of the superimposed area in the depth map
    if int(depth_map.shape[0] * superimpose_ratio) > overlay_depth_map.shape[0]:
        overlay_row = overlay_depth_map.shape[0]
    else:
        overlay_row = int(depth_map.shape[0] * superimpose_ratio)

    if int(depth_map.shape[1] * superimpose_ratio) > overlay_depth_map.shape[1]:
        overlay_column = overlay_depth_map.shape[1]
    else:
        overlay_column = int(depth_map.shape[1] * superimpose_ratio)

    # Modify depth map, corresponding to the color image modifications
    if variation == 1:  # Top average out
        depth_map[: int(depth_map.shape[0] * average_out_ratio), :] = mean_depth
    elif variation == 2:  # Bottom average out
        depth_map[- int(depth_map.shape[0] * average_out_ratio):, :] = mean_depth
    elif variation == 3:  # Left average out
        depth_map[:, : int(depth_map.shape[1] * average_out_ratio)] = mean_depth
    elif variation == 4:  # Right average out
        depth_map[:, - int(depth_map.shape[1] * average_out_ratio):] = mean_depth
    elif variation == 7:  # Superimpose top left with bottom right of overlay depth map
        depth_map[: overlay_row, : overlay_column] = overlay_depth_map[- overlay_row:, - overlay_column:]
    elif variation == 8:  # Superimpose top right with bottom left of overlay depth map
        depth_map[: overlay_row, - overlay_column:] = overlay_depth_map[- overlay_row:, : overlay_column]
    elif variation == 9:  # Superimpose bottom left with top right of overlay depth map
        depth_map[- overlay_row:, : overlay_column] = overlay_depth_map[: overlay_row, - overlay_column:]
    elif variation == 10:  # Superimpose bottom right with top left of overlay depth map
        depth_map[- overlay_row:, - overlay_column:] = overlay_depth_map[: overlay_row, : overlay_column]

    return depth_map
