import cv2
import numpy as np
from matplotlib import pyplot as plt
import heapq


def export_quads_to_obj(quads, file_name):
    """
    Exports all the quads to a single OBJ file.
    """
    with open(file_name, 'w') as file:
        vertex_count = 1
        for quad in quads:
            #print("quad=", quad)
            if len(quad) < 4:
                print("Not a quad (in export), skipping")
                continue
            for v in quad:
                #print("v=", v)
                file.write(f"v {v[0]} {v[1]} 0\n")  # Z coordinate is 0 as we are dealing with 2D data
            file.write(f"f {vertex_count} {vertex_count + 1} {vertex_count + 2} {vertex_count + 3}\n")
            vertex_count += 4


def compute_tight_bounding_quadrilateral(contour_vertices):
    points = np.array(contour_vertices, dtype='float32')
    hull = cv2.convexHull(points, returnPoints=True).squeeze()

    if len(hull) <= 4:
        return hull

    while len(hull) > 4:
        min_area_increase = float('inf')
        min_area_index = -1

        for i in range(len(hull)):
            p1 = hull[i - 1]
            p2 = hull[i]
            p3 = hull[(i + 1) % len(hull)]

            area = calculate_added_area(p1, p2, p3)
            if area < min_area_increase:
                min_area_increase = area
                min_area_index = i

        hull = np.delete(hull, min_area_index, axis=0)

    #hull.append(hull[0])  # Close the hull
    return hull

# Function to calculate area formed by three points
def calculate_added_area(p1, p2, p3):
    # Using Shoelace formula
    return 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p2[0]*p1[1] - p3[0]*p2[1] - p1[0]*p3[1])


def render_convex_hull_to_image(convex_hull):
    # Calculate the axis-aligned bounding box of the convex hull
    x, y, w, h = cv2.boundingRect(convex_hull)

    # Create a blank image without a buffer around the bounding box
    img = np.zeros((h, w), dtype=np.uint8)

    # Adjust convex hull points to the new image coordinate system
    adjusted_hull = convex_hull - np.array([[x, y]])

    # Draw the convex hull on the image
    cv2.drawContours(img, [adjusted_hull], -1, (255, 255, 255), 1)

    # Return the image and the top-left corner of the AABB as offset
    return img, (x, y)

def find_dense_points_from_image(img, aabb_offset, buffer=10):
    # Add padding to the image
    padded_img = np.pad(img, ((buffer, buffer), (buffer, buffer)), mode='constant', constant_values=0)

    # Find contours which returns the dense points
    contours, _ = cv2.findContours(padded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        # Remove the buffer and add the AABB offset to each contour point
        dense_points = np.vstack(contours).squeeze()
        dense_points -= buffer  # Removing the buffer
        dense_points += np.array(aabb_offset)  # Adding the AABB offset
        return dense_points
    return np.array([])

def validateHull(dense_hull):
    max_distances = []
    for i in range(len(dense_hull)):
        p = dense_hull[i - 1]
        n = dense_hull[(i + 1) % len(dense_hull)]
        c = dense_hull[i]

        # Calculating Euclidean distances
        dist_cp = np.linalg.norm(c - p)
        dist_cn = np.linalg.norm(c - n)

        # Find the maximum distance
        max_dist = max(dist_cp, dist_cn)
        max_distances.append(max_dist)

    #for i, dist in enumerate(max_distances):
    #    print(f"Point {i}: Max distance to neighbors = {dist}")

    # If you want to print the overall maximum
    overall_max = max(max_distances)
    #print(f"Overall maximum distance between neighboring points: {overall_max}")
    assert overall_max < 1.42

def refine_corners(approx_quad, convex_hull, num_iterations=25):
    refined_quad = approx_quad.copy()
    approx_quad = approx_quad.copy()

    if len(approx_quad) != 4:
        print("Not a quadrilateral")
        return approx_quad

    hull_image, aabb_offs = render_convex_hull_to_image(convex_hull)
    dense_convex_hull = find_dense_points_from_image(hull_image, aabb_offs)
    #validateHull(dense_convex_hull)
    for ri in range(num_iterations):
        for i in range(len(approx_quad)):
            mid_point = (approx_quad[(i + 1) % 4] + approx_quad[(i - 1) % 4]) / 2
            direction_vector_mid = mid_point - approx_quad[i]
            direction_vector_adj1 = approx_quad[(i + 1) % 4] - approx_quad[i]
            direction_vector_adj2 = approx_quad[(i - 1) % 4] - approx_quad[i]

            # Normalize the direction vectors
            direction_vector_mid /= np.linalg.norm(direction_vector_mid)
            direction_vector_adj1 /= np.linalg.norm(direction_vector_adj1)
            direction_vector_adj2 /= np.linalg.norm(direction_vector_adj2)

            distances = np.linalg.norm(dense_convex_hull - approx_quad[i], axis=1)
            closest_point_index = np.argmin(distances)
            closest_point = dense_convex_hull[closest_point_index]
            #print("currently at:", approx_quad[i])
            #print("closest point:", closest_point)
            #print("len(dense_convex_hull)=", len(dense_convex_hull))

            possible_moves = [dense_convex_hull[closest_point_index - 1],
                              dense_convex_hull[closest_point_index],
                              dense_convex_hull[(closest_point_index + 1) % len(dense_convex_hull)]]

            best_move_score = float('-inf')
            best_move = None
            for move in possible_moves:
                move_direction = move - approx_quad[i]
                move_direction /= np.linalg.norm(move_direction)

                scores = []
                for direction_vector in [direction_vector_mid, direction_vector_adj1, direction_vector_adj2]:
                    cosine_similarity = np.dot(move_direction, direction_vector)
                    # Apply a smooth exponential J-Curve for scoring
                    score = np.exp(-cosine_similarity * 10)
                    scores.append(score)

                total_score = sum(scores)
                if total_score > best_move_score:
                    best_move_score = total_score
                    best_move = move

            refined_quad[i] = best_move

        if np.array_equal(approx_quad, refined_quad):
            print("** NO CHANGE IN QUAD **")
            break
        else:
            approx_quad = refined_quad.copy()

    return refined_quad


def upscale_and_process_image(image_path):
    all_refined_quads = []

    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Upscale the image by 4x using bi-cubic interpolation
    upscale_size = (inverted_image.shape[1]*4, inverted_image.shape[0]*4)
    upscaled_image = cv2.resize(inverted_image, upscale_size, interpolation=cv2.INTER_CUBIC)

    # Apply a gentle Gaussian blur
    blurred_image = cv2.GaussianBlur(upscaled_image, (5, 5), 0)

    # Apply threshold to binarize the image
    _, thresholded_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image again
    final_background = 255 - thresholded_image

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw false-color connected components on the final background image
    output_image = cv2.cvtColor(final_background, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        # Compute the convex hull of the contour
        convex_hull = cv2.convexHull(contour)

        # Draw random color contours
        #cv2.drawContours(output_image, [convex_hull], -1, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 1)
        
        # Draw the convex hull
        #cv2.polylines(output_image, [np.array(convex_hull)], True, (0, 255, 0), 1)

        # Approximate best fit quadrilateral
        approx_quad = compute_tight_bounding_quadrilateral(convex_hull)
        # Refine the corners of the approximated quadrilateral
        refined_quad = refine_corners(approx_quad, convex_hull)
        all_refined_quads.append(refined_quad)
        #print("**************************************")
        #print("* appro_quad=", approx_quad)
        #print("* refined_quad=", refined_quad)
        #print("**************************************")
        # Convert refined_quad to integer for drawing
        refined_quad_int = np.array(refined_quad, dtype=np.int32).reshape((-1, 1, 2))
        # Ensure points are integer and in the correct shape for polylines
        approx_quad_int = np.array(approx_quad, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the approximated quadrilateral
        #cv2.polylines(output_image, [approx_quad_int], True, (0, 0, 255), 1)
        # Draw the refined quadrilateral in a different color
        cv2.polylines(output_image, [refined_quad_int], True, (255, 0, 0), 1)

        #if i > 1000: break

    export_quads_to_obj(all_refined_quads, '/mnt/c/Users/joe/Documents/Assets/all_refined_quads.obj')

    # Display the images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB)), plt.title('Thresholded Upscaled Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), plt.title('Output with Contours and Convex Hulls')
    plt.show()

    # Save the output image to disk
    cv2.imwrite('output_with_contours_and_hulls.png', output_image)

# Example usage
upscale_and_process_image('cape_guy_two_view.png')

