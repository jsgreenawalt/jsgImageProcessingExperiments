import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import bezier
import skimage.measure

def load_image_and_extract_outline(image_path):
    # Original code to load an image and extract its outline remains unchanged
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    image_array = (image_array > 128).astype(np.uint8)
    contours = skimage.measure.find_contours(image_array, 0.5)
    return contours[0]

def optimize_segment(p0, p3, segment):
    # Function to optimize a bezier curve segment
    def objective_function(ctrl_pts):
        ctrl_pts_full = np.vstack([p0, ctrl_pts.reshape(2, 2), p3])
        curve = bezier.Curve(np.array(ctrl_pts_full).T, degree=3)
        distances = [np.linalg.norm(curve.evaluate(s).flatten() - point) for s, point in zip(np.linspace(0, 1, len(segment)), segment)]
        return sum(distances)

    p1 = p0 + (p3 - p0) / 3
    p2 = p0 + 2 * (p3 - p0) / 3
    guess = np.array([p1, p2])
    result = minimize(objective_function, guess.flatten(), method='L-BFGS-B')
    error = result.fun
    optimized_p1, optimized_p2 = result.x.reshape(2, 2)
    return error, optimized_p1, optimized_p2

def bezier_curve_fitting(points, error_threshold):
    control_points = []
    segment_start = 0  # Index of the start of the current segment

    while segment_start < len(points) - 1:
        # Start with a segment of length 2 (minimum for a bezier curve)
        segment_end = segment_start + 1

        while segment_end < len(points):
            segment = np.array(points[segment_start:segment_end + 1])
            p0, p3 = points[segment_start], points[segment_end]
            error, optimized_p1, optimized_p2 = optimize_segment(p0, p3, segment)
            print("error=", error)
            if error > error_threshold:
                print('new segment')
                #if segment_end == segment_start + 1:
                #    # Segment is too small to split; accept the current error
                #    break
                #else:
                if True:
                    # Revert to the last acceptable segment
                    segment_end -= 1
                    segment = np.array(points[segment_start:segment_end + 1])
                    p0, p3 = points[segment_start], points[segment_end]
                    _, optimized_p1, optimized_p2 = optimize_segment(p0, p3, segment)
                    break

            segment_end += 1

        # Add the control points of the segment to the list
        control_points.append(np.vstack([p0, optimized_p1, optimized_p2, p3]))
        segment_start = segment_end  # Start new segment from the end of the last segment

    return control_points

def plot_curve(control_points_list, points):
    # Original plot_curve function remains unchanged
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], linestyle='-', color='blue', label='Outline')

    for segment_number, cp in enumerate(control_points_list, start=1):
        curve = bezier.Curve(cp.T, degree=3)
        s_vals = np.linspace(0, 1, 256)
        curve_points = curve.evaluate_multi(s_vals).T
        ax.plot(curve_points[:, 0], curve_points[:, 1], color='green')

        for point_number, point in enumerate(cp, start=1):
            ax.scatter(*point, color='red')
            ax.annotate(f'S{segment_number}P{point_number}', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.legend()
    return fig

def main():
    image_path = 'binary_bezier_fit_test_1.png'
    points = load_image_and_extract_outline(image_path)
    error_threshold = 30.0  # Set an appropriate error threshold for curve fitting
    control_points_list = bezier_curve_fitting(points, error_threshold)
    fig = plot_curve(control_points_list, points)

    output_image_path = 'binary_bezier_fit_test_output.png'
    fig.savefig(output_image_path)
    print("Image saved to", output_image_path)

    for i, cp in enumerate(control_points_list, 1):
        print(f"Control Points for Segment {i}:")
        print(cp)

if __name__ == "__main__":
    main()

