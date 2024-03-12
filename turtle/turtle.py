#!/usr/bin/python3.11
import pygame
import random
import math
import numpy as np
import cv2
import time

# Constants for layout
GUI_WIDTH = 200  # Adjust width as necessary
THUMBNAIL_WIDTH = 600  # Adjust width as necessary
THUMBNAIL_HEIGHT = 600  # Adjust height as necessary
THUMBNAIL_SCALE = 15  # Scale factor for thumbnails

# globals
gui_rect = None
input_image_rect = None
thumbnails_area_rect = None

def numpy_array_to_pygame_image(img_data):
    """
    Converts a numpy array of normalized pixel values into a Pygame image.

    :param img_data: A numpy array of shape (height, width, 3) with normalized pixel values.
    :return: A Pygame surface representing the image.
    """
    # First, denormalize the pixel values (convert them back to 0-255 range)
    denormalized_data = (img_data * 255).astype(np.uint8)

    # Create an empty Pygame surface with the same dimensions
    height, width, _ = denormalized_data.shape
    pygame_image = pygame.Surface((width, height))

    # Iterate over each pixel and copy the color to the Pygame surface
    for y in range(height):
        for x in range(width):
            color = denormalized_data[y, x, :]  # Extract the RGB color
            pygame_image.set_at((x, y), color)

    return pygame_image

def normalize_angle(angle):
    """ Normalize the angle to be within the range of 0 to 2π radians. """
    return angle % (2 * math.pi)

def points_to_angle(start_point, end_point):
    """
    Calculate the angle in radians of the vector defined by two points with respect to the positive x-axis.

    :param start_point: A numpy array of shape (2,) representing the start point of the vector.
    :param end_point: A numpy array of shape (2,) representing the end point of the vector.
    :return: The normalized angle in radians between the positive x-axis and the vector.
    """
    # Calculate the vector from the start point to the end point
    vector = end_point - start_point
    
    # Calculate the angle of the vector
    angle = np.arctan2(vector[1], vector[0])
    
    return angle

def find_longest_arc(radarSamples):
    # we need to treat this as a circular buffer -- we don't stop when we get to the end of the 
    # array, we stop when we get back to another arc that we've already processed
    rayNum = 0
    lastThresholdState = True
    stop = False
    all_arcs = []
    while True:
        for sample in radarSamples[rayNum]:
            turtlePos = sample["turtlePos"]
            sampleValue = sample["sampleValue"]
            samplePos = sample["samplePos"]
            is_above_threshold = sample["overThreshold"]
            try: x = sample["visited"]
            except: sample["visited"] = False

            if lastThresholdState and not is_above_threshold:
                # new arc
                if sample["visited"]:
                    # we already added this one, we can stop now.
                    stop = True
                    break
                # make  a new arc and add the current sample
                all_arcs.append([sample])

            elif not lastThresholdState and not is_above_threshold:
                # arc continues, add to current arc
                all_arcs[-1].append(sample)

            # set for next iteration
            lastThresholdState = is_above_threshold
            sample["visited"] = True

        if stop: break

        # set for next iteration
        rayNum += 1
        rayNum = rayNum % len(radarSamples)

    if len(all_arcs) == 0:
        print("whoopsie")

    # Find the largest arc
    longestArc = all_arcs[0]
    for arc in all_arcs:
        arcLen = len(arc)
        if arcLen > len(longestArc):
            longestArc = arc
    return longestArc

    

def find_arc_center(arc):
    startSample = arc[0]
    endSample = arc[-1]
    startAngle = points_to_angle(startSample["turtlePos"], startSample["samplePos"])
    endAngle = points_to_angle(endSample["turtlePos"], endSample["samplePos"])
    if endAngle < startAngle:
        # If the arc crosses the 0-radian mark, adjust the end angle by adding 2π
        endAngle += 2 * math.pi
    centerAngle = ((endAngle - startAngle) / 2) + startAngle
    centerAngle = centerAngle % (2 * math.pi)
    return centerAngle


# utility methods and data storage/conversion
class TurtleUtil:
    def __init__(self, image, mask, mark, screen, image_size):
        self.normalized_speed = 1 / (max(image_size) * 4)  # Speed in normalized space
        self.image = image
        self.mask = mask
        self.mark = mark
        self.color_threshold = 0.1
        self.screen = screen
        self.image_size = image_size

        # Convert Pygame image to numpy array, normalize pixel values, and transpose axes
        self.img_data = pygame.surfarray.array3d(self.image).astype(np.float32).transpose((1, 0, 2)) / 255
        # Convert Pygame image to numpy array and normalize pixel values
        #self.img_data = pygame.surfarray.array3d(self.image).astype(np.float32) / 255
        testImage = numpy_array_to_pygame_image(self.img_data)
        screen.blit(testImage, (0,0))
        
        self.sweep_angle_increment = 5
        self.samples_per_sweep = 1

        self.sample_history = []  # Stores the last 10 samples
        self.sample_averages = []  # Stores averages of every 10 samples
        self.cumulative_average = [0, 0, 0]  # Initialize cumulative average (assuming RGB values)
        self.total_samples = 0  # Total number of samples taken

    def update_sample_history(self, new_sample):
        # Append new sample
        if len(self.sample_history) >= 10:
            # Roll off the oldest sample when the history is full
            self.sample_history.pop(0)
        self.sample_history.append(new_sample)

        self.total_samples += 1

        # Update cumulative average
        for i in range(3):  # Assuming RGB values
            self.cumulative_average[i] = \
                ((self.cumulative_average[i] * (self.total_samples - 1))\
                + new_sample["sampleValue"][i]) / self.total_samples
        #print("cumulative_average=", self.cumulative_average)
        '''
        # Store the average every 10 samples
        if self.total_samples % 10 == 0:
            average_sample = [sum(col) / 10 for col in zip(*self.sample_history)]
            self.sample_averages.append(average_sample)

            # Keep only the last 10 averages
            if len(self.sample_averages) > 10:
                self.sample_averages.pop(0)
        '''

    def calc_local_unit_vec(self, angle):
        vx = math.cos(angle) * (self.normalized_speed * 4.0)
        vy = math.sin(angle) * (self.normalized_speed * 4.0)
        return vx, vy

    def sample_continuous_field(self, x, y, turtlePosition, update_sample_history=False):
        # Map the normalized coordinates to actual image coordinates
        x_img = (x * self.image_size[1]) - 1
        y_img = (y * self.image_size[0]) - 1

        # Clamp the coordinates to the image size
        x_img = max(0, min(x_img, self.image_size[1] - 1))
        y_img = max(0, min(y_img, self.image_size[0] - 1))

        # Find the integer coordinates of the surrounding pixels
        x0, y0 = int(x_img), int(y_img)
        x1, y1 = min(x0 + 1, self.image_size[1] - 1), min(y0 + 1, self.image_size[0] - 1)

        # Calculate the weights for each pixel
        x_weight = x_img - x0
        y_weight = y_img - y0

        # Get the values of the four surrounding pixels
        top_left = self.img_data[y0, x0]
        top_right = self.img_data[y0, x1]
        bottom_left = self.img_data[y1, x0]
        bottom_right = self.img_data[y1, x1]

        # Compute the weighted average
        top = top_left * (1 - x_weight) + top_right * x_weight
        bottom = bottom_left * (1 - x_weight) + bottom_right * x_weight
        interpolated_value = top * (1 - y_weight) + bottom * y_weight

        turtlePos = np.array(turtlePosition)
        sampleValue = np.array(interpolated_value)
        samplePos = np.array((x, y))
        is_above_threshold = any(abs(sampleValue[i] - self.cumulative_average[i]) > self.color_threshold for i in range(3))
        sample =\
        {
                "turtlePos": turtlePos,
                "sampleValue": sampleValue,
                "samplePos": samplePos,
                "overThreshold": is_above_threshold
        }
        # Update the sample history
        if update_sample_history:
            self.update_sample_history(sample)

        return sample
    
    def reflect_angle(self, angle):
        # Reflects the angle as if bouncing off an edge
        # This can be adjusted to change the behavior of the bounce
        return normalize_angle(math.pi - angle if random.random() < 0.5 else -angle)

    def sample_line(self, turtlePosition, direction, length, update_sample_history=False):
        samples = []
        positions = []
        x, y = turtlePosition
        for i in range(length):
            x += direction[0]
            y += direction[1]
            sample = self.sample_continuous_field(x, y, turtlePosition, update_sample_history)
            samples.append(sample)
        return samples

    def radar_sweep(self, position):
        all_samples = []
        for angle_degree in range(0, 360, self.sweep_angle_increment):
            angle_radian = math.radians(angle_degree)
            direction_vector = self.calc_local_unit_vec(angle_radian)
            line_samples = self.sample_line(position, direction_vector, self.samples_per_sweep)
            all_samples.append(line_samples)
        return all_samples


    def create_thumbnail(self, center, scale, angle):
        # Map normalized coordinates to image coordinates
        center_x = int(center[0] * self.image.get_width())
        center_y = int(center[1] * self.image.get_height())

        # Define the size of the area to capture around the turtle
        half_scale = scale // 2

        # Calculate the top-left corner of the thumbnail
        start_x = max(0, center_x - half_scale)
        start_y = max(0, center_y - half_scale)

        # Ensure the thumbnail does not exceed the image boundaries
        end_x = min(self.image.get_width(), start_x + scale)
        end_y = min(self.image.get_height(), start_y + scale)

        # Adjust the size if it goes beyond the image boundaries
        capture_width = end_x - start_x
        capture_height = end_y - start_y

        # Create a Rect for the thumbnail area
        thumbnail_rect = pygame.Rect(start_x, start_y, capture_width, capture_height)

        # Extract the thumbnail area from the image
        thumbnail = self.image.subsurface(thumbnail_rect).copy()

        # Scale the thumbnail
        thumbnail_surf = pygame.transform.scale(thumbnail,
                (THUMBNAIL_WIDTH // 2, THUMBNAIL_HEIGHT // 2))

        # Center coordinates for the line
        center_x = thumbnail_surf.get_width() // 2
        center_y = thumbnail_surf.get_height() // 2

        # Calculate the end coordinates for the line based on the angle and speed
        # Scale the length as per thumbnail scale
        line_length = self.normalized_speed * scale * max(self.image_size[0], self.image_size[1])
        end_x = center_x + line_length * math.cos(angle) * 4.0
        end_y = center_y + line_length * math.sin(angle) * 4.0

        # Draw the line on the thumbnail
        #pygame.draw.line(thumbnail_surf, (255, 0, 0), (center_x, center_y), (end_x, end_y), 2)
        #pygame.draw.circle(thumbnail_surf, (255, 0, 0), (center_x, center_y), 3)

         # Get radar sweep samples and positions
        is_above_threshold = False
        radar_samples = self.radar_sweep(center)
        for line_samples in radar_samples:
            for sample in line_samples:
                turtlePos = sample["turtlePos"]
                sampleValue = sample["sampleValue"]
                samplePos = sample["samplePos"]
                is_above_threshold = sample["overThreshold"]
                sampleVec = (samplePos - turtlePos) * scale * max(self.image_size[0], self.image_size[1])
                # Map normalized position to thumbnail scale

                # Translate relative position to thumbnail coordinates
                lineStart = (center_x, center_y)
                lineEnd = (center_x, center_y) + (sampleVec * 4.0)

                # Determine color based on threshold
                if is_above_threshold:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # Draw line for this sample
                pygame.draw.line(thumbnail_surf, color, lineStart, lineEnd, 1)

        # Draw the line on the thumbnail
        pygame.draw.line(thumbnail_surf, (255, 0, 0), (center_x, center_y), (end_x, end_y), 2)
        pygame.draw.circle(thumbnail_surf, (255, 0, 0), (center_x, center_y), 3)

        return thumbnail_surf, is_above_threshold

# behaviors expose five methods, sense, think, mark, move, and introspect
# sense -- examine surrounding environment (ex, radar_sweep)
# think -- perform analysis and decide results (new heading, marker state)
# mark -- mark a boundary, region, or both at the given location
# move -- move to a new location based on the rules of this behavior
# introspect -- if the behavior is to continue, returns self, otherwise ???
class BehaviorAvoidEdges():
    def __init__(self, turtleObj, tu):
        self.tu = tu
        self.turtleObj = turtleObj

    def sense(self):
        # Get current position and direction of the turtle
        current_position = self.turtleObj.normalized_position
        current_direction = self.tu.calc_local_unit_vec(self.turtleObj.angle)

        # Determine the length of the sample line
        # This can be a fixed value or dynamically calculated based on some criteria
        # for now, we're looking only 1 pixel-dimension-equivalent ahead
        sample_length = 1

        # Sample along the line in the current direction, this gets stored in history
        # for later examination
        sample = self.tu.sample_line(current_position, current_direction, sample_length, update_sample_history=True)[0]
        deltas = [abs(sample["sampleValue"][i] - self.tu.cumulative_average[i]) for i in range(3)]
        if any(delta > self.tu.color_threshold for delta in deltas):
            print("edge detected")
            self.turtleObj.edge_detected = True
        else:
            self.turtleObj.edge_detected = False
        return self.turtleObj.edge_detected

    def think(self):
        edge_threshold = self.tu.color_threshold
        # Check the latest sample against the threshold
        current_sample = self.tu.sample_history[-1]
        deltas = [abs(current_sample["sampleValue"][i] - self.tu.cumulative_average[i]) for i in range(3)]

        # Check if any of the deltas exceed the threshold
        if any(delta > edge_threshold for delta in deltas):
            print("edge detected")
            # Edge is detected
            self.turtleObj.edge_detected = True

            # Perform radar sweep to examine the surroundings
            radarSamples = self.tu.radar_sweep(self.turtleObj.normalized_position)
            longestArc = find_longest_arc(radarSamples)

            # Calculate the center angle of the largest arc
            center_angle = find_arc_center(longestArc)

            # Set turtle's angle to the center of the largest arc
            self.turtleObj.angle = normalize_angle(center_angle)
            print("new angle(6)=", self.turtleObj.angle)
            print("self.tu.cumulative_average=", self.tu.cumulative_average)
        else:
            # No edge detected, continue in the current direction
            self.turtleObj.edge_detected = False
        return self.turtleObj.edge_detected

    def mark(self):
        pass

    def move(self):
        nx, ny = self.turtleObj.normalized_position
        # Calculate velocity components based on angle and normalized speed
        vx, vy = self.tu.calc_local_unit_vec(self.turtleObj.angle)
        vx = math.cos(self.turtleObj.angle) * self.tu.normalized_speed
        vy = math.sin(self.turtleObj.angle) * self.tu.normalized_speed

        nx += vx
        ny += vy
        # Boundary check and angle adjustment
        if nx < 0 or nx > 1:
            # Reflect angle horizontally
            self.turtleObj.angle = normalize_angle(math.pi - self.turtleObj.angle)
            nx = max(0, min(nx, 1))
            self.turtleObj.angle = normalize_angle(self.turtleObj.angle)
            print("new angle(2)=", self.turtleObj.angle)
        if ny < 0 or ny > 1:
            self.turtleObj.angle = normalize_angle(-self.turtleObj.angle)  # Reflect angle vertically
            ny = max(0, min(ny, 1))
            self.turtleObj.angle = normalize_angle(self.turtleObj.angle)
            print("new angle(3)=", self.turtleObj.angle)
        self.turtleObj.normalized_position = (nx, ny)

    def introspect(self):
        # must return current object if this behavior is maintaining control
        return self








# Define the Turtle class
class Turtle():
    def __init__(self, image, mask, mark, screen, image_size):
        self.tu = TurtleUtil(image, mask, mark, screen, image_size)

        self.normalized_position = (random.random(), random.random())
        self.normalized_position = (0.42,0.4)
        self.angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        print("new angle(4)=", self.angle)

        self.image = image
        self.mask = mask
        self.mark = mask
        self.screen = screen
        self.edge_detected = False # Flag for edge detection
        self.behavior_avoid_edges = BehaviorAvoidEdges(self, self.tu)
        self.current_behavior = self.behavior_avoid_edges

    def tick(self):
        # change behavior?
        self.current_behavior = self.current_behavior.introspect()
        # mark current location
        self.current_behavior.mark()
        # we move based on decision reached in the *prior* tick, because reasons
        self.current_behavior.move()
        # what's going on around
        pause = self.current_behavior.sense()
        # think over the next move 
        self.current_behavior.think()
        # pause for a sec?
        return pause

    def visualize(self):
        # Convert normalized position to screen coordinates
        x, y = self.normalized_position
        # Take into account the GUI width offset
        screen_x = int(x * self.tu.image_size[0]) + GUI_WIDTH
        screen_y = int(y * self.tu.image_size[1])

        # Change color based on edge detection
        color = (255, 0, 0) if self.edge_detected else (0, 255, 0)
        # Draw the turtle ensuring it's within the bounds of the input image area
        if input_image_rect.collidepoint(screen_x, screen_y):
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 5)

        # Reset edge detection flag after visualization
        #self.edge_detected = False

# entry point
def main():
    global gui_rect, input_image_rect, thumbnails_area_rect
    # Pygame setup
    pygame.display.init()
    pygame.font.init()
    
    # Load an image and set screen size dynamically
    image = pygame.image.load('input_image.png')  # Replace with actual image path
    image_size = image.get_size()

    # Calculate total screen size
    screen_width = image_size[0] + GUI_WIDTH + THUMBNAIL_WIDTH
    screen_height = image_size[1]
    screen_size = (screen_width, screen_height)

    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Ear Detection Prototype")

    # Define screen regions correctly
    gui_rect = pygame.Rect(0, 0, GUI_WIDTH, screen_height)
    input_image_rect = pygame.Rect(GUI_WIDTH, 0, image_size[0], image_size[1])
    thumbnails_area_rect = pygame.Rect(GUI_WIDTH + image_size[0], 0, THUMBNAIL_WIDTH, screen_height)

    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Ear Detection Prototype")

    mask = None  # Placeholder for the actual mask
    mark = None  # Placeholder for the actual mask

    # Create turtles in normalized space
    num_turtles = 1
    turtles = []
    for i in range(0, num_turtles):
        turtles.append(Turtle(image, mask, mark, screen, image_size))

   # Set up a clock for tick-based simulation
    clock = pygame.time.Clock()
    ticks_per_second = 10000

    # Main loop
    running = True
    pause = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drawing
        screen.fill((255, 255, 255))

        # Draw the GUI controls area (gray box)
        pygame.draw.rect(screen, (200, 200, 200), gui_rect)

        # Draw the input image area (black box)
        pygame.draw.rect(screen, (0, 0, 0), input_image_rect)
        screen.blit(image, input_image_rect.topleft)

        # Draw and update the turtles in the input image area
        for turtle in turtles:
            pause = turtle.tick()
            turtle.visualize()

       # Draw the thumbnail area background (gray box for the entire area)
        pygame.draw.rect(screen, (200, 200, 200), thumbnails_area_rect)

        thumbnail_size = (THUMBNAIL_WIDTH // 2, THUMBNAIL_HEIGHT // 2)
        for i, turtle in enumerate(turtles):
            # Calculate the thumbnail grid position
            cols = THUMBNAIL_WIDTH // thumbnail_size[0]
            row = i // cols
            col = i % cols
            thumb_x = thumbnails_area_rect.x + (col * thumbnail_size[0])
            thumb_y = thumbnails_area_rect.y + (row * thumbnail_size[1])

            # Create and draw thumbnail for each turtle
            thumb_center = turtle.normalized_position
            thumb_surf, _ = turtle.tu.create_thumbnail(thumb_center, THUMBNAIL_SCALE, turtle.angle)
            screen.blit(thumb_surf, (thumb_x, thumb_y))

        # Flip the display
        pygame.display.flip()
        if pause:
            print("paused")
            ticks_per_second = 1
        else: ticks_per_second = 10000
        clock.tick(ticks_per_second)

    pygame.quit()


if __name__ == "__main__":
    main()



