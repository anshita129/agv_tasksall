import pygame
import numpy as np  # Efficient array manipulations for merging

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class MapMerger:
    def __init__(self, width, height):
        """Initialize the merged map with appropriate dimensions."""
        self.width = width
        self.height = height

    def merge_maps(self, map1, map2):
        """
        Merges two maps by combining their explored regions.
        - White areas indicate explored regions.
        - Black areas indicate unexplored regions.
        """
        if not isinstance(map1, pygame.Surface) or not isinstance(map2, pygame.Surface):
            raise TypeError("Both map1 and map2 must be Pygame Surface objects.")
        # Create a merged map
        merged_map = pygame.Surface((self.width, self.height))
        merged_map.fill(BLACK)

        # Convert maps to numpy arrays for easier pixel manipulation
        map1_array = pygame.surfarray.array3d(map1)
        map2_array = pygame.surfarray.array3d(map2)
        merged_array = np.maximum(map1_array, map2_array)

        # Convert back to pygame Surface
        pygame.surfarray.blit_array(merged_map, merged_array)
        return merged_map

    def align_maps(self, map1, map2, displacement):
        """
        Aligns map2 with map1 using a calculated displacement (x, y).
        Useful when agents start in unknown relative positions.
        """
        aligned_map = pygame.Surface((self.width, self.height))
        aligned_map.fill(BLACK)

        # Blit map1 directly
        aligned_map.blit(map1, (0, 0))

        # Blit map2 shifted by the calculated displacement
        aligned_map.blit(map2, displacement)
        return aligned_map
