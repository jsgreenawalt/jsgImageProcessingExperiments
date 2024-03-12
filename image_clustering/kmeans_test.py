import cv2
import nevergrad as ng
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_high_contrast_colormap(n_colors):
    hsv_colors = [(i/n_colors, 1, 1) for i in range(1, n_colors)]  # Skip 0 to avoid black
    rgb_colors = [mcolors.hsv_to_rgb(c) for c in hsv_colors]
    rgb_colors.insert(0, (0, 0, 0))
    rgb_colors.insert(1, (1, 1, 1))
    return mcolors.ListedColormap(rgb_colors)

def cluster_image(image_path, num_clusters=2, boundary_shift=0.5, display_images=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cmap = create_high_contrast_colormap(1024)

    # Adding x and y coordinates as channels
    height, width, _ = image.shape
    x_coords = np.tile(np.linspace(0, 1, width), (height, 1))
    y_coords = np.tile(np.linspace(0, 1, height), (width, 1)).T
    extended_pixels = np.dstack((image, x_coords, y_coords)).reshape((-1, 5))

    kmeans = KMeans(n_clusters=num_clusters, tol=0.00001, random_state=1, n_init=10)
    kmeans.fit(extended_pixels)
    distances = kmeans.transform(extended_pixels)
    ratio = distances[:, 0] / (distances[:, 0] + distances[:, 1])
    adjusted_ratio = ratio - (0.5 - boundary_shift)
    reclassified_labels = np.where(adjusted_ratio < 0.5, 0, 1)
    reclassified_image = reclassified_labels.reshape(image.shape[:2]) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(reclassified_image.astype(np.uint8), connectivity=8)
    num_components = num_labels - 1

    proportion_class_1 = np.mean(reclassified_labels == 0)
    proportion_class_2 = 1 - proportion_class_1

    if display_images:
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 4, 1)
        plt.title('Original Image')
        plt.imshow(image, interpolation='none')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Clustered Image')
        plt.imshow(kmeans.labels_.reshape(image.shape[:2]), cmap=cmap, interpolation='none')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Reclassified Image')
        plt.imshow(reclassified_image, cmap='gray', interpolation='none')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Connected Components')
        plt.imshow(labels, cmap=cmap, interpolation='none')
        plt.axis('off')

        plt.show()

    print(f"The original image has {num_components} disconnected components.")
    print(f"The reclassified image has {num_labels - 1} disconnected components.")
    return num_labels - 1, proportion_class_1, proportion_class_2

def optimize_boundary_shift(image_path, num_clusters=2):
    def objective(boundary_shift):
        num_components, prop_class_1, prop_class_2 = cluster_image(image_path, num_clusters, boundary_shift, display_images=False)
        if num_components <= 0 or prop_class_1 > 0.85 or prop_class_2 > 0.85:
            return float('inf')
        return num_components

    parametrization = ng.p.Scalar(lower=0, upper=1)
    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(objective)
    optimized_boundary_shift = recommendation.value
    return optimized_boundary_shift

image_path = './female_anime_realistic_3heads__00425_.png'
optimized_shift = optimize_boundary_shift(image_path)
print(f"Optimized boundary_shift: {optimized_shift}")
cluster_image(image_path, boundary_shift=optimized_shift, display_images=True)

