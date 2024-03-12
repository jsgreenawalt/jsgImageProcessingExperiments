import matplotlib.pyplot as plt
from skimage import io, transform, filters  # Imported filters for Gaussian blur
from skimage.color import rgb2gray
from skimage.measure import find_contours
import numpy as np
import pygame
import time
from sklearn.cluster import KMeans
from math import sqrt

# Existing function to load and prepare the image
def load_and_prepare_image(image_path):
    image = io.imread(image_path)
    # Applying Gaussian blur to the image
    # Modification: Added Gaussian blur to smooth the image before further processing
    image = filters.gaussian(image, sigma=1)  # Gaussian blur with a sigma of 1
    if len(image.shape) == 3:
        image = rgb2gray(image)
    return image

# Existing function to upscale the image
def upscale_image(image, scale_factor):
    upscaled_image = transform.rescale(image, scale_factor, anti_aliasing=False, mode='reflect', order=1)
    if len(upscaled_image.shape) == 3:
        upscaled_image = np.mean(upscaled_image, axis=2)
    return upscaled_image

# Existing function to initialize Pygame
def initialize_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption('Contours Animation')
    return screen

# Existing function to scale image to fit the window size
def scale_image_to_window(image, window_size):
    height, width = image.shape
    scale_x = window_size[0] / width
    scale_y = window_size[1] / height
    scale_factor = min(scale_x, scale_y)
    scaled_size = (int(width * scale_factor), int(height * scale_factor))
    scaled_image = transform.resize(image, scaled_size, anti_aliasing=True)
    return scaled_image, scale_factor

# Existing function to prepare image for Pygame display
def prepare_image_for_pygame(image):
    scaled_image_pygame = (image * 255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(np.stack([scaled_image_pygame]*3, axis=-1).swapaxes(0, 1))
    return surface

def box_count(img, k):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                           np.arange(0, img.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z):
    # Ensure Z is a binary image
    assert(len(Z.shape) == 2)
    assert(Z.dtype == bool or Z.dtype == np.bool_)

    # Calculating the fractal dimension
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))

    # Customized progression of box sizes for increased sensitivity
    sizes = np.geomspace(1, n, num=30, endpoint=True).astype(int)  # Geometric progression
    sizes = np.unique(sizes)  # Ensure unique sizes

    counts = [box_count(Z, size) for size in sizes]

    # Linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return abs(-coeffs[0])

def calculate_fractal_dimension_from_contour(contour, padding=5):
    """
    Calculate the fractal dimension of a contour using the box-counting method.
    """
    # Bounds of the contour
    min_y, min_x = np.min(contour, axis=0) - padding
    max_y, max_x = np.max(contour, axis=0) + padding
    size = (int(max_y - min_y), int(max_x - min_x))

    # Create a binary image
    image = np.zeros(size, dtype=bool)
    for y, x in contour:
        x, y = int(x) - int(min_x), int(y) - int(min_y)
        image[y, x] = True

    # Calculate the fractal dimension
    fd = fractal_dimension(image)

    #print("Binary Image of the Contour Dim:" + str(fd))
    # Display the binary image
    #plt.imshow(image, cmap='gray')
    #plt.title("Binary Image of the Contour Dim:" + str(fd))
    #plt.show()

    return fd

def calculate_area_score(contour):
    """
    Calculate the normalized area score of a contour.

    Parameters:
    - contour: A numpy array of contour points.

    Returns:
    - Normalized area score.
    """
    area = np.abs(np.dot(contour[:, 0], np.roll(contour[:, 1], 1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], 1))) / 2
    normalized_area = area / 9437184  # Normalize the area
    return normalized_area

def calculate_roughness_score(contour):
    """
    Calculate the roughness score of a contour based on its fractal dimension.

    Parameters:
    - contour: A numpy array of contour points.

    Returns:
    - Roughness score.
    """
    roughness = calculate_fractal_dimension_from_contour(contour)
    return roughness

def compound_score(contour, area_weight):
    """
    Calculate a compound score based on area and roughness.

    Parameters:
    - contour: A numpy array of contour points.
    - area_weight: Weight for the area in the final score calculation.

    Returns:
    - Compound score.
    """
    area_score = calculate_area_score(contour)
    roughness_score = calculate_roughness_score(contour)
    if roughness_score == 0:
        roughness_score = 0.001

    score = area_score + (1.0 / sqrt(roughness_score))
    return score, area_score, roughness_score

# Function to calculate bounding box of a contour
def calculate_bbox(contour):
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    return np.array([x_min, y_min, x_max, y_max])

# Function to cluster bounding boxes using k-means
def cluster_bounding_boxes(bboxes, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=15)
    kmeans.fit(bboxes)
    return kmeans.labels_, kmeans.cluster_centers_

# Function to determine if one bbox is within another
def bbox_within(bbox1, bbox2):
    return np.all(bbox1[:2] >= bbox2[:2]) and np.all(bbox1[2:] <= bbox2[2:])

# Function to find hierarchy in clustered bounding boxes
def find_bbox_hierarchy(cluster_labels, cluster_centers, bboxes):
    hierarchy = [[] for _ in range(len(cluster_centers))]
    for idx, (label, bbox) in enumerate(zip(cluster_labels, bboxes)):
        for other_idx, center in enumerate(cluster_centers):
            if bbox_within(bbox, center) and label != other_idx:
                hierarchy[other_idx].append(idx)
    return hierarchy

# Modified main function to incorporate new logic
def main():
    image_path = "./sdxl_binary_mask.png"
    #image_path = "./sdxl_binary_mask_4.png"
    #image_path = "./sdxl_deet_test_1.png"
    window_size = (2048, 2048)

    image = load_and_prepare_image(image_path)
    upscaled_image = upscale_image(image, 3)
    screen = initialize_pygame(window_size)
    scaled_image, scale_factor = scale_image_to_window(upscaled_image, window_size)
    surface = prepare_image_for_pygame(scaled_image)

    running = True
    paused = False
    level = 0.0
    area_weight = 0.0010
    n_clusters = 10  # Number of clusters for k-means
    large_contours = []  # List to store large contours

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            level += 0.01
            if level > 1.0:
                paused = True

            # Collect contours with area greater than 100
            contours = find_contours(upscaled_image, level)
            for contour in contours:
                score, area, roughness = compound_score(contour, area_weight)
                if area >= 0.005 and roughness < 1.00:
                    print("score, area, roughness=", score, area, roughness)
                    large_contours.append(contour)

            image_rect = surface.get_rect(center=screen.get_rect().center)
            # Pause to cluster and render after collecting contours
            if paused:
                # Cluster bounding boxes
                bboxes = np.array([calculate_bbox(contour) for contour in large_contours])
                cluster_labels, cluster_centers = cluster_bounding_boxes(bboxes, n_clusters)
                hierarchy = find_bbox_hierarchy(cluster_labels, cluster_centers, bboxes)

                best_contours = [None] * len(cluster_centers)
                for contour, bbox, label in zip(large_contours, bboxes, cluster_labels):
                    score, _, _ = compound_score(contour, area_weight)
                    if best_contours[label] is None or score > compound_score(best_contours[label], area_weight)[0]:
                        best_contours[label] = contour

                # Debugging output for hierarchy and metrics
                for i, cluster in enumerate(hierarchy):
                    print(f"Cluster {i}: Center = {cluster_centers[i]}, Contours = {len(cluster)}")
                    for contour_index in cluster:
                        contour = large_contours[contour_index]
                        score, area, roughness = compound_score(contour, area_weight)
                        bbox = bboxes[contour_index]
                        print(f"  Contour {contour_index}: Score = {score}, Area = {area}, Roughness = {roughness}, BBox = {bbox}")

                # Draw the best contour for each cluster
                screen.fill((0, 0, 0))
                screen.blit(surface, image_rect)

                for contour in best_contours:
                    if contour is not None:
                        scaled_contour = contour * scale_factor
                        scaled_contour += np.array([image_rect.left, image_rect.top])
                        pygame.draw.lines(screen, (0, 255, 0), False, scaled_contour[:, [1, 0]], 2)
                pygame.display.flip()
                time.sleep(0.01)
            else:
                for contour in large_contours:
                    if contour is not None:
                        scaled_contour = contour * scale_factor
                        scaled_contour += np.array([image_rect.left, image_rect.top])
                        pygame.draw.lines(screen, (255, 0, 0), False, scaled_contour[:, [1, 0]], 2)
                pygame.display.flip()

    pygame.quit()

# Call the main function
if __name__ == "__main__":
    main()
