import pygame
import math
from robot_api import RobotAPI  # Import the existing RobotAPI class

WHITE = (255, 255, 255)

class Localization:
    def __init__(self, world_width, world_height):
        self.map = pygame.Surface((world_width, world_height)) 
        self.map.fill((0, 0, 0))  # Initially mark everything unexplored (black)
        self.world_height = world_height
        self.world_width = world_width

    def update(self, agent):
        """Update the map using agent's scan data."""
        sensor_data = agent.scan()  # Get LIDAR data (distance to walls)
        heading = agent.get_imu_data()  # Get agent’s heading
        start_x, start_y = agent.get_pos()  # Current position of the agent

        fov = 360
        resolution = 2
        start_angle = heading - fov / 2.0

        for i, distance in enumerate(sensor_data):
            ray_angle = start_angle + i * resolution
            ray_angle_rad = math.radians(ray_angle)

            end_x = int(start_x + distance * math.cos(ray_angle_rad))
            end_y = int(start_y + distance * math.sin(ray_angle_rad))

            # Draw explored area as white
            pygame.draw.line(self.map, WHITE, (start_x, start_y), (end_x, end_y), 1)
            if distance < agent.__max_range:
                pygame.draw.circle(self.map, WHITE, (end_x, end_y), 2)  # Mark endpoint

    def at(self, x, y):
        """Check the state of a given point in the map (explored or unexplored)."""
        return self.map.get_at((int(x), int(y)))
    
    def get_size(self):
        return self.map.get_size()

