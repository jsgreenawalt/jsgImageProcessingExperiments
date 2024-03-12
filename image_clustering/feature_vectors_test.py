import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import random
import math

def load_image(image_path):
    """ Load an image as both PIL and HSV tensor. """
    # Load the original image
    original_image = Image.open(image_path)

    # Convert to HSV and then to a tensor
    hsv_image = original_image.convert('HSV')
    transform = transforms.Compose([transforms.ToTensor()])
    hsv_tensor = transform(hsv_image).float()

    return original_image, hsv_tensor

def compute_gradient_vectors(hsv_tensor):
    """ Compute gradient vectors for each HSV channel. """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3)

    gradient_vectors = []

    for i in range(3):  # Iterate over H, S, and V channels
        channel = hsv_tensor[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        grad_x = torch.nn.functional.conv2d(channel, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(channel, sobel_y, padding=1)

        gradient_vector = torch.cat((grad_x, grad_y), dim=1).squeeze(0)
        gradient_vectors.append(gradient_vector)

    return gradient_vectors

def scale_image(image, scale_factor):
    """ Scale up an image by a specified factor with hard-square pixels. """
    width, height = image.size
    return image.resize((width * scale_factor, height * scale_factor), Image.NEAREST)

def draw_gradient_vectors(image, gradient_vectors, scale_factor, is_smoothed=False):
    """ Draw gradient vectors on the scaled image. """
    colors = [(255, 0, 0, 85), (0, 255, 0, 85), (0, 0, 255, 85)]  # RGBA for Red, Green, Blue with alpha
    final_image = image.convert("RGBA")

    for i, color in enumerate(colors):
        # Create a transparent overlay for each HSV channel
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for y in range(gradient_vectors[i].shape[1]):
            for x in range(gradient_vectors[i].shape[2]):
                grad_x = gradient_vectors[i][0, y, x].item()
                grad_y = gradient_vectors[i][1, y, x].item()

                start_x = x * scale_factor + scale_factor // 2
                start_y = y * scale_factor + scale_factor // 2

                end_x = start_x + grad_x * scale_factor
                end_y = start_y + grad_y * scale_factor

                draw.line([start_x, start_y, end_x, end_y], fill=color, width=1)
                draw.rectangle([start_x - 1, start_y - 1, start_x + 1, start_y + 1], fill=color)
        # Composite the overlay onto the final image
        final_image = Image.alpha_composite(final_image, overlay)

    return final_image.convert('RGB')  # Convert back to RGB if needed

class VehicleSimulator:
    """
    Encapsulates vehicle simulation functionality.
    Modified to prevent the vehicle from revisiting pixels.
    """
    def __init__(self, gradient_vectors, image_size):
        self.gradient_vectors = gradient_vectors
        self.image_size = image_size
        self.momentum = [0, 0]  # Initial momentum is zero
        self.visited = set()  # Track visited pixels
        self.position = self.initialize_vehicle()
        self.movement_mode = []  # Track movement mode for each move

    def initialize_vehicle(self):
        """ Initialize the vehicle at a random position and mark it as visited. """
        initial_position = [random.randint(0, self.image_size[0] - 1), random.randint(0, self.image_size[1] - 1)]
        self.visited.add(tuple(initial_position))  # Mark the initial position as visited
        return initial_position

    def calculate_reward(self, x, y):
        """ Calculate the reward based on gradient vector magnitude. """
        magnitude = sum(np.linalg.norm([self.gradient_vectors[i][0, y, x].item(), self.gradient_vectors[i][1, y, x].item()]) for i in range(3))
        return magnitude

    def get_neighbors(self, x, y):
        """ Get the 8-way neighboring pixels. """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.image_size[0] and 0 <= ny < self.image_size[1]:
                    neighbors.append((nx, ny))
        return neighbors

    def angle_between_vectors(self, v1, v2):
        """ Calculate the angle between two vectors. """
        dot_product = np.dot(v1, v2)
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot_product / magnitude_product)
        return angle

    def count_visited_neighbors(self, x, y):
        """ Count the number of visited neighbors for a given pixel. """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.visited:
                    count += 1
        return count

    def move_vehicle(self):
        print("Position=", self.position)
        neighbors = self.get_neighbors(self.position[0], self.position[1])
        best_reward = -1
        best_position = self.position
        penalty_factor = 0.01  # Adjust this factor to balance the penalty

        for nx, ny in neighbors:
            if (nx, ny) in self.visited:
                continue

            # Check if moving to nx, ny would result in more than one visited neighbor
            if self.count_visited_neighbors(nx, ny) > 2:
                continue

            reward = self.calculate_reward(nx, ny)

            # Calculate direction penalty
            new_direction = [nx - self.position[0], ny - self.position[1]]
            if any(self.momentum) and any(new_direction):  # Avoid division by zero
                angle_change = self.angle_between_vectors(self.momentum, new_direction)
                direction_penalty = math.degrees(angle_change) * penalty_factor
                reward -= direction_penalty

            if reward > best_reward:
                best_reward = reward
                best_position = [nx, ny]

        if best_reward < 0.35:  # threshold for random walk
            print("random walking")
            current_mode = 'random'  # Set movement mode to random walk
            while True:
                angle = random.uniform(0, 2 * np.pi)
                self.momentum = [np.cos(angle) * 10.0, np.sin(angle) * 10.0]
                best_position = [self.position[0] + int(np.round(self.momentum[0])),
                                 self.position[1] + int(np.round(self.momentum[1]))]
                if tuple(best_position) not in self.visited:  # Check if the new position is not visited
                    break
        else:
            current_mode = 'normal'  # Set movement mode to random walk
            print("normal movement, best_reward=", best_reward)
            self.momentum = [best_position[0] - self.position[0], best_position[1] - self.position[1]]

        # Ensure new position is within bounds
        best_position[0] = max(0, min(best_position[0], self.image_size[0] - 1))
        best_position[1] = max(0, min(best_position[1], self.image_size[1] - 1))
        self.position = best_position
        self.movement_mode.append(current_mode)  # Record movement mode
        self.visited.add(tuple(self.position))  # Mark the new position as visited

    def simulate_vehicle(self, moves=20000):
        """ Simulate the vehicle movement and render the trail. """
        vehicle_path = []
        for _ in range(moves):
            vehicle_path.append(self.position)
            self.move_vehicle()
        return vehicle_path

    def render_vehicle_trail(self, vehicle_path, image):
        """ Render the vehicle trail on the image. """
        trail_image = image.convert("RGBA")
        draw = ImageDraw.Draw(trail_image)
        for idx, position in enumerate(vehicle_path):
            color = (255, 0, 0) if self.movement_mode[idx] == 'random' else (0, 255, 0)  # Red for random walk, green for normal
            draw.point(position, fill=color)
        return trail_image

if __name__ == "__main__":
    image_path = "./sdxl_binary_mask.png"
    scale_factor = 10

    original_image, hsv_tensor = load_image(image_path)
    gradient_vectors = compute_gradient_vectors(hsv_tensor)
    scaled_image = scale_image(original_image, scale_factor)
    result_image = draw_gradient_vectors(scaled_image, gradient_vectors, scale_factor)
    result_image.save("gradient_visualization.png")

    vehicle_simulator = VehicleSimulator(gradient_vectors, original_image.size)
    vehicle_path = vehicle_simulator.simulate_vehicle()
    trail_image = vehicle_simulator.render_vehicle_trail(vehicle_path, original_image)
    trail_image.save("vehicle_trail.png")

