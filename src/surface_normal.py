import numpy as np
import cv2 as cv


def compute_surface_normal(test_number, tile_number, random_test):
    # Read the file
    if (test_number % 30) >= 15:
        filename = "./TestImages/" + str(int(test_number / 30)) + "/Front_Down.npy"
        random_filename = "./TestImages/" + str(random_test) + "/Front_Down.npy"
    else:
        filename = "./TestImages/" + str(int(test_number / 30)) + "/Front_Up.npy"
        random_filename = "./TestImages/" + str(random_test) + "/Front_Up.npy"

    # Load the numpy file
    depth_image = np.load(filename)
    random_depth_image = np.load(random_filename)

    # Modify the depth image based on the alterations in the rgb image
    variation = test_number % 15
    mean_depth = np.mean(depth_image)

    whiteout_blackout_ratio = 0.5
    superimpose_ratio = 0.5

    if int(depth_image.shape[0] * superimpose_ratio) > random_depth_image.shape[0]:
        half_row = random_depth_image.shape[0]
    else:
        half_row = int(depth_image.shape[0] * superimpose_ratio)

    if int(depth_image.shape[1] * superimpose_ratio) > random_depth_image.shape[1]:
        half_col = random_depth_image.shape[1]
    else:
        half_col = int(depth_image.shape[1] * superimpose_ratio)

    if variation == 1:  # Top whiteout
        depth_image[: int(depth_image.shape[0] * whiteout_blackout_ratio), :] = mean_depth
    elif variation == 2:  # Bottom whiteout
        depth_image[int(depth_image.shape[0] * (1 - whiteout_blackout_ratio)):, :] = mean_depth
    elif variation == 3:  # Left blackout
        depth_image[:, : int(depth_image.shape[1] * whiteout_blackout_ratio)] = mean_depth
    elif variation == 4:  # Right blackout
        depth_image[:, int(depth_image.shape[1] * (1 - whiteout_blackout_ratio)):] = mean_depth
    elif variation == 7:  # Superimpose, top left
        depth_image[: half_row, : half_col] = random_depth_image[random_depth_image.shape[0] - half_row:, random_depth_image.shape[1] - half_col:]
    elif variation == 8:  # Superimpose, top right
        depth_image[: half_row, depth_image.shape[1] - half_col:] = random_depth_image[random_depth_image.shape[0] - half_row:, : half_col]
    elif variation == 9:  # Superimpose, bottom left
        depth_image[depth_image.shape[0] - half_row:, : half_col] = random_depth_image[: half_row, random_depth_image.shape[1] - half_col:]
    elif variation == 10:  # Superimpose, bottom right
        depth_image[depth_image.shape[0] - half_row:, depth_image.shape[1] - half_col:] = random_depth_image[: half_row, : half_col]

    # Select the tile
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    if tile_number == 1:
        tile = depth_image[: int(height / 3), : int(width / 3)]
    elif tile_number == 2:
        tile = depth_image[: int(height / 3), int(width / 3): int(2 * width / 3)]
    elif tile_number == 3:
        tile = depth_image[: int(height / 3), int(2 * width / 3):]
    elif tile_number == 4:
        tile = depth_image[int(height / 3): int(2 * height / 3), : int(width / 3)]
    elif tile_number == 5:
        tile = depth_image[int(height / 3): int(2 * height / 3), int(2 * width / 3):]

    # Replace any nan values with the maximum value in the tile
    max_value = np.max(tile)
    tile[np.where(((tile == 0) | (tile == np.nan)))] = max_value

    # Segment the depth tile via Otsu's threshold
    tile_image = np.floor((tile / max_value) * 255).astype(np.uint8)
    # cv.imshow("original", tile_image)
    # cv.waitKey(0)

    _, segmented_image = cv.threshold(tile_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # cv.imshow("segmented", segmented_image)
    # cv.waitKey(0)

    # Compute the surface normal at any point and normalize
    # normal_image = np.zeros_like(tile)
    count = 0
    z_normal_sum = 0
    dummy_normilize = np.zeros(3)
    for i in range(1, tile.shape[0] - 1):
        for j in range(1, tile.shape[1] - 1):
            if segmented_image[i, j] == 255:
                dzdx = tile[i + 1, j] - tile[i - 1, j]
                dzdy = tile[i, j + 1] - tile[i, j - 1]
                pixel_surface_normal = cv.normalize(np.array([-dzdx, -dzdy, 0.01]), dummy_normilize)
                z_normal_sum += pixel_surface_normal[2]
                count += 1

            # normal_image[i,j] = pixel_surface_normal[2]

    # Take the average of surface normals in the tile
    # and compute the dot product with the z-axis by choosing only the z component of the surface normals
    if count > 0:
        average_normal_z = z_normal_sum / count
    else:
        average_normal_z = 0

    # normal_image_conv = np.floor((normal_image / np.max(normal_image)) * 255).astype(np.uint8)
    # cv.imshow("normal", normal_image_conv)
    # cv.waitKey(0)
    # print(average_normal_z)

    return average_normal_z
