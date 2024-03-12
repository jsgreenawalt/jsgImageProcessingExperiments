import cv2
import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import math

def gimp_brightness_contrast_map(value, brightness, slant):
    # Apply brightness adjustment
    if brightness < 0.0:
        value = value * (1.0 + brightness)
    else:
        value = value + ((1.0 - value) * brightness)

    # Apply contrast adjustment (slant)
    value = (value - 0.5) * slant + 0.5

    return value

def gimp_brightness_contrast_adjustment(image, brightness, contrast):
    # Scale brightness from GIMP's range [-127, 127] to [-1, 1] and then divide by 2 as per GIMP's implementation
    brightness = (brightness / 127.0) / 2.0

    # Scale contrast from GIMP's range [-127, 127] to [-1, 1] and then adjust for slant calculation
    contrast = (contrast / 127.0)

    # Calculate the slant for contrast adjustment
    slant = np.tan((contrast + 1) * np.pi / 4)

    # Normalize image to range [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # Apply the brightness and contrast adjustments
    adjusted_image = np.vectorize(gimp_brightness_contrast_map)(
        normalized_image, brightness, slant)

    # Clip and convert back to [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 1) * 255.0
    adjusted_image = adjusted_image.astype(np.uint8)

    return adjusted_image

################

def create_high_contrast_colormap(n_colors):
    hsv_colors = [(i/n_colors, 1, 1) for i in range(1, n_colors)]  # Skip 0 to avoid black
    rgb_colors = [mcolors.hsv_to_rgb(c) for c in hsv_colors]
    rgb_colors.insert(0, (0, 0, 0))
    rgb_colors.insert(1, (1, 1, 1))
    return mcolors.ListedColormap(rgb_colors)

jsgCMAP = create_high_contrast_colormap(1024)

def apply_sobel_and_normalize(channel, ksize=5, amount=1.0):
    """
    Apply Sobel filter and normalize the output with an 'amount' factor to enhance edge detection.
    This is akin to the 'amount' slider in GIMP's Sobel edge detection, controlling the edge visibility.

    Args:
    channel: Image channel (R, G, or B) to apply the Sobel filter.
    ksize (int): Kernel size for the Sobel filter. Default is 5.
    amount (float): Factor to scale the Sobel output. Default is 2.0.

    Returns:
    numpy.ndarray: 8-bit image after applying Sobel filter and normalization.
    """
    # Apply Sobel filter
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.hypot(sobelx, sobely)

    # Scale the Sobel output by the 'amount' factor
    sobel_scaled = sobel * amount

    # Normalize and convert to 8-bit format
    sobel_normalized = cv2.normalize(sobel_scaled, None, 0, 255, cv2.NORM_MINMAX)
    sobel_normalized_8bit = np.uint8(sobel_normalized)

    return sobel_normalized_8bit

def optimize_threshold(image, original_image):
    def objective(threshold):
        num_components, masks = segment_image(image, original_image, threshold, display_images=False)
        #print("threshold=", threshold, "Connected_Components=", num_components)
        #if num_components < 2: numComponents = float('inf')
        return num_components

    parametrization = ng.p.Scalar(lower=1, upper=127)
    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=250)
    recommendation = optimizer.minimize(objective)
    optimized_threshold = recommendation.value
    return optimized_threshold

def segment_image_preprocess(image_path):
    image = cv2.imread(image_path)
    originalImage = image
    image = gimp_brightness_contrast_adjustment(image, 100.0, 100.0)
    image = gimp_brightness_contrast_adjustment(image, 100.0, 100.0)
    image = gimp_brightness_contrast_adjustment(image, 100.0, 100.0)
    R, G, B = cv2.split(image)
    R_sobel = apply_sobel_and_normalize(R)
    G_sobel = apply_sobel_and_normalize(G)
    B_sobel = apply_sobel_and_normalize(B)
    sobel_combined = cv2.merge([R_sobel, G_sobel, B_sobel])
    sobel_gray = cv2.cvtColor(sobel_combined, cv2.COLOR_RGB2GRAY)
    return sobel_gray, image, originalImage

def segment_image(sobel_gray, original_image, threshold=3, display_images=True):
    _, binary = cv2.threshold(sobel_gray, threshold, 255, cv2.THRESH_BINARY)
    binary_8bit = np.uint8(binary)
    binary_for_floodfill = np.copy(binary_8bit)
    h, w = binary_for_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(binary_for_floodfill, mask, (0,0), 255)
    xor_result = cv2.bitwise_xor(binary_8bit, binary_for_floodfill)

    # Use cv2.connectedComponentsWithStats to get stats, centroids and labels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(xor_result), connectivity=8)

    connected_regions = num_labels - 1  # Subtracting one to exclude the background label

    # Create masks for each connected component
    component_masks = [labels == i for i in range(1, num_labels)]  # Exclude the background label (0)

    if display_images:
        plt.figure(figsize=(12, 8))
        titles = ['Original Image', 'Combined Sobel', 'Binary', 'XOR Result', "Inverted", "Labeled"]
        images = [original_image, sobel_gray, binary_8bit, xor_result, cv2.bitwise_not(xor_result), labels]
        cmaps = ['gray', 'gray', 'gray', 'gray', 'gray', jsgCMAP]

        for i in range(len(titles)):
            plt.subplot(2, 4, i + 1)
            if i == 0:
                plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(images[i], cmap=cmaps[i], interpolation='none')
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    #print("Connected Components:", connected_regions)
    if connected_regions < 2: 
        return float('inf'), []

    # Return the number of connected components and the masks
    return connected_regions, component_masks


def combine_masks_and_get_inverse(masks):
    """
    Combine multiple masks into a single mask and return the inverse of the combined mask.

    Args:
    masks (list of numpy.ndarray): List of boolean masks.

    Returns:
    combined_mask (numpy.ndarray): The combined mask.
    inverse_mask (numpy.ndarray): The inverse of the combined mask.
    """
    if not masks:
        raise ValueError("The list of masks is empty")

    # Initialize the combined mask with the first mask
    combined_mask = masks[0].copy()

    # Combine the rest of the masks
    for mask in masks[1:]:
        combined_mask = np.logical_or(combined_mask, mask)

    # Generate the inverse of the combined mask
    inverse_mask = np.logical_not(combined_mask)

    return combined_mask, inverse_mask

#################
def create_background_mask(image):
    """
    Create a mask to identify non-background pixels in the image.
    
    Args:
    image: Image array to create a mask for.
    
    Returns:
    mask: A boolean mask where True represents non-background pixels.
    """
    # Create a mask where all non-black pixels are marked as True
    mask = np.any(image != [0, 0, 0], axis=-1)

    return mask

def find_dominant_clusters_masked(image, mask, n_clusters=3):
    """
    Apply K-means clustering to find dominant color clusters in a masked image.
    
    Args:
    image: Image array in which to find clusters.
    mask: Boolean mask to exclude background pixels.
    n_clusters (int): Number of clusters to form.
    
    Returns:
    labels: The labels of clusters in the original image size.
    centers: The centers of clusters.
    """
    # Apply mask to the image and reshape for clustering
    masked_image = image[mask]
    reshaped_image = masked_image.reshape((-1, 3))

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(reshaped_image)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Expand labels back to the size of the original image
    full_labels = np.zeros(image.shape[:2], dtype=int) - 1  # Initialize with -1 for background
    full_labels[mask] = labels

    return full_labels, centers

def generate_child_masks(labels, dominant_label, parent_mask):
    """
    Generate masks for child segments based on the clustering results.

    Args:
    labels: Array of cluster labels.
    dominant_label: The label of the dominant cluster.
    parent_mask: The mask of the parent segment.

    Returns:
    dominant_mask: Mask for the dominant cluster segment.
    other_mask: Mask for the 'everything else' segment.
    """
    labels_reshaped = labels.reshape(parent_mask.shape)
    dominant_mask = (labels_reshaped == dominant_label) & parent_mask
    other_mask = ~dominant_mask & parent_mask
    return dominant_mask, other_mask

# Modification: New helper function to apply a mask to an image segment
def apply_mask_to_segment(image, mask):
    """
    Apply a mask to an image segment.

    Args:
    image: The image to mask.
    mask: The mask to apply.

    Returns:
    masked_segment: The image segment with the mask applied.
    """
    return np.where(mask[:, :, None], image, 0)



def segment_clusters(image, labels, dominant_label):
    """
    Segment the image into two regions based on the dominant cluster.

    Args:
    image: Image array to segment.
    labels: Cluster labels for each pixel.
    dominant_label: The label of the dominant cluster.

    Returns:
    dominant_segment: Image segment of the dominant cluster.
    other_segment: Image segment of all other clusters combined.
    """
    # Reshaping labels array to match image dimensions
    labels_reshaped = labels.reshape(image.shape[:2])

    # Creating masks for segments
    dominant_mask = labels_reshaped == dominant_label
    other_mask = ~dominant_mask

    # Creating segments
    dominant_segment = np.where(dominant_mask[:,:,None], image, 0)
    other_segment = np.where(other_mask[:,:,None], image, 0)

    return dominant_segment, other_segment

def analyze_mask_distribution(mask):
    total_pixels = mask.size
    true_pixels = np.sum(mask)
    percentage = (true_pixels / total_pixels) * 100
    return percentage

def optimize_brightness_contrast(image, mask, optimize_budget=10):
    """
    Optimize brightness and contrast parameters for a given image segment.

    Args:
    image: The image containing the segment.
    mask: The mask defining the segment.
    optimize_budget: The budget for the optimization process.

    Returns:
    (optimized_brightness, optimized_contrast): Tuple of optimized parameters.
    """
    def objective(params):
        brightness, contrast = params
        adjusted_image = gimp_brightness_contrast_adjustment(image, brightness, contrast)
        masked_image = apply_mask_to_segment(adjusted_image, mask)

        # Apply clustering to the adjusted segment
        labels, _ = find_dominant_clusters_masked(masked_image, mask, n_clusters=3)
        
        # Count unique labels (clusters) - ignoring background (-1)
        unique_labels = np.unique(labels[labels >= 0])
        return len(unique_labels)  # Minimize this number

    parametrization = ng.p.Tuple(ng.p.Scalar(lower=-100, upper=100), ng.p.Scalar(lower=-100, upper=100))
    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=optimize_budget)
    recommendation = optimizer.minimize(objective)
    return recommendation.value

def recursive_subdivision(image, mask, depth, mask_tree, level=0):
    # Debugging: Print the current recursion level and mask status
    print(f"Recursion Level: {level}, Mask sum: {np.sum(mask)}")

    # Base case: No further subdivision needed
    if depth == 0 or np.sum(mask) == 0:
        return

    # Optimize brightness and contrast for the current segment
    optimized_brightness, optimized_contrast = optimize_brightness_contrast(image, mask)
    adjusted_segment = gimp_brightness_contrast_adjustment(image, optimized_brightness, optimized_contrast)
    masked_adjusted_segment = apply_mask_to_segment(adjusted_segment, mask)

    labels, centers = find_dominant_clusters_masked(masked_adjusted_segment, mask)
    valid_labels = labels[mask]
    if len(valid_labels) > 0:
        dominant_label = np.argmax(np.bincount(valid_labels))
    else:
        return

    dominant_mask, other_mask = generate_child_masks(labels, dominant_label, mask)

    # Analyze mask distribution
    dominant_percentage = analyze_mask_distribution(dominant_mask)
    other_percentage = analyze_mask_distribution(other_mask)
    print(f"Level {level}: Dominant Mask %: {dominant_percentage:.2f}%, Other Mask %: {other_percentage:.2f}%")

    # Ensure the current level exists in mask_tree
    while len(mask_tree) <= level:
        mask_tree.append([])

    # Append newly created masks to the current level
    mask_tree[level].append(dominant_mask)
    mask_tree[level].append(other_mask)

    # Recursive calls for further subdivision
    recursive_subdivision(image, dominant_mask, depth - 1, mask_tree, level + 1)
    recursive_subdivision(image, other_mask, depth - 1, mask_tree, level + 1)

    # Debugging: Print current structure of mask_tree
    print(f"After Level {level} Recursion - Mask Tree Structure:")
    for lvl, masks in enumerate(mask_tree):
        print(f"  Level {lvl}: {len(masks)} masks")
        for i, m in enumerate(masks):
            print(f"    Mask {i}: {np.sum(m)} pixels")

# Revised display_segments function for displaying binary masks
def display_segments(mask_tree):
    """
    Display the binary masks from a tree of masks.

    Args:
    mask_tree: Tree of masks.
    """
    plt.figure(figsize=(12, 8))
    for level, masks in enumerate(mask_tree):
        for i, mask in enumerate(masks):
            plt.subplot(len(mask_tree), len(masks), level * len(masks) + i + 1)
            plt.imshow(mask, cmap='gray')  # Display mask in grayscale
            plt.title(f'Level {level + 1} - Mask {i + 1}')
            plt.axis('off')
    # Additional debug prints
    print("Displaying Masks:")
    for level, masks in enumerate(mask_tree):
        for i, mask in enumerate(masks):
            print(f"Level {level + 1} - Mask {i + 1}: {np.sum(mask)} pixels")
    plt.show()

#####################


def main():
    '''
    #image_path = './female_anime_realistic_3heads__00425_.png'
    #image_path = './ComfyUI_temp_trkto_00007_.png'
    image_path = './sdxl_binary_mask_4.png'
    sobel_gray, adjusted_image, original_image = segment_image_preprocess(image_path)
    optimized_threshold = optimize_threshold(sobel_gray, adjusted_image)
    print(f"Optimized threshold: {optimized_threshold}")
    numRegions, masks = segment_image(sobel_gray, adjusted_image, threshold=optimized_threshold)
    foreground_mask, background_mask = combine_masks_and_get_inverse(masks)
    '''

    #################
    # TODO:
    # now we're done separating from background, here comes the next part
    # we need to reduce the remaining masked areas one-at-a-time, dividing
    # each area into two regions.  The criteria for division are as follows:
    # 1) cluster by color
    # 2) find the cluster with the most members
    # 3) our split is "that cluster" and "everything else"
    # In this way, we find the dominant region, and "everything else"
    # Next issue is that the results are not "good", ideally, there should be no holes
    # or islands in the dominant region or the "everything else" region -- we can
    # optimize the cluster boundaries with nevergrad until we get the desired results
    # Please adhere to all code modification guidelines.
    
    # New code for the TODO implementation
    image_path = './sdxl_binary_mask_4.png'
    sobel_gray, adjusted_image, original_image = segment_image_preprocess(image_path)
    optimized_threshold = optimize_threshold(sobel_gray, original_image)
    print(f"Optimized threshold: {optimized_threshold}")
    numRegions, masks = segment_image(sobel_gray, original_image, threshold=optimized_threshold)
    
    foreground_mask, background_mask = combine_masks_and_get_inverse(masks)

    # Modified call to recursive_subdivision with the initial mask
    #segments = recursive_subdivision(original_image, foreground_mask, depth=3)
    mask_tree = []
    segments = recursive_subdivision(original_image, foreground_mask, depth=3, mask_tree=mask_tree)
    display_segments(mask_tree)  # Display only the image segments, not the masks

if __name__=="__main__":
    main()
