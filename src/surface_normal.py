import numpy as np
import cv2 as cv


def compute_average_surface_normal_depth(depth_map):
    """
    Compute average length of the object surface normal's image on z-axis (depth axis) for all the pixels that lay on the object (z-axis is the perpendicular to the image plane).
    :param depth_map: Input the depth map
    :return: Average depth of the surface normal for all the pixels on the object surface (float)
    """

    # Replace any nan values with the maximum value in the tile (thus set them as background)
    max_depth = np.max(depth_map)
    depth_map[np.where(((depth_map == 0) | (depth_map == np.nan)))] = max_depth

    # Segment the depth map via Otsu threshold method
    converted_depth_map = np.floor((depth_map / max_depth) * 255).astype(np.uint8)
    _, segmented_map = cv.threshold(converted_depth_map, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # For any point on object surface (foreground)
    foreground_count = normal_depth_sum = 0
    for i in range(1, depth_map.shape[0] - 1):
        for j in range(1, depth_map.shape[1] - 1):
            if segmented_map[i, j] == 255:

                # Compute horizontal and vertical depth derivatives
                dzdx = depth_map[i + 1, j] - depth_map[i - 1, j]
                dzdy = depth_map[i, j + 1] - depth_map[i, j - 1]

                # Compute the surface normal
                pixel_surface_normal = cv.normalize(np.array([dzdx, dzdy, 1]), dst=None)

                # Dot product with the z-axis (depth axis) by choosing only the z (depth) component of the surface normal
                normal_depth_sum += pixel_surface_normal[2]

                # Keep account of the foreground pixels
                foreground_count += 1

    # Take the average of surface normals in the depth map
    if foreground_count > 0:
        average_normal_z = normal_depth_sum / foreground_count
    else:
        average_normal_z = 0

    return average_normal_z


def get_depth_map(test_number, random_test):
    """
    Read the frontal and random depth maps.
    :param test_number: The current test number (int)
    :param random_test: Random test to superimpose part of the front image with the random one

    :return: Frontal depth map, random test map
    """

    # Find the file addresses to read
    if (test_number % 30) >= 15:
        frontal_filename = "./test_set/" + str(int(test_number / 30)) + "/Front_Down.npy"
        random_filename = "./test_set/" + str(random_test) + "/Front_Down.npy"
    else:
        frontal_filename = "./test_set/" + str(int(test_number / 30)) + "/Front_Up.npy"
        random_filename = "./test_set/" + str(random_test) + "/Front_Up.npy"

    # Load the depth maps
    frontal_depth_image = np.load(frontal_filename)
    random_depth_image = np.load(random_filename)

    return frontal_depth_image, random_depth_image
