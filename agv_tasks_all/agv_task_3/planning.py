import pygame
import heapq
import math
import random

class Planner:
    def not_very_close(self, map_surface, coordinates, map_width, map_height, buffer_zone=5):
        SAFE_COLOR = (255, 255, 255)
        BARRIER_COLOR = (181, 101, 29)

        # Verify if the given coordinates are within the map boundaries
        x_pos, y_pos = int(coordinates[0]), int(coordinates[1])
        if x_pos < 0 or x_pos >= map_width or y_pos < 0 or y_pos >= map_height:
            return False

        try:
            if map_surface.get_at((x_pos, y_pos)) != SAFE_COLOR:
                return False
        except IndexError:
            return False

        # Examine the surrounding area within the buffer zone
        for offset_x in range(-buffer_zone, buffer_zone + 1):
            for offset_y in range(-buffer_zone, buffer_zone + 1):
                nearby_x, nearby_y = x_pos + offset_x, y_pos + offset_y

                # Skip points that are out of map boundaries
                if (nearby_x < 0 or nearby_x >= map_width or 
                    nearby_y < 0 or nearby_y >= map_height):
                    continue

                # Check if the point is a barrier
                try:
                    nearby_pixel_color = map_surface.get_at((nearby_x, nearby_y))
                    if nearby_pixel_color[:3] == BARRIER_COLOR:
                        return False
                except IndexError:
                    continue

        # The position is deemed safe
        return True


    def get_path(self, surface, world_height, world_width, first_pos, last_pos):
        first_pos = (int(first_pos[0]), int(first_pos[1]))
        last_pos = (int(last_pos[0]), int(last_pos[1]))
        
        
        WHITE = (255, 255, 255)
        buffer_zone = 5
    
        try:
            if first_pos[0] < 0 or first_pos[0] >= world_width or first_pos[1] < 0 or first_pos[1] >= world_height:
                print(f"Invalid first_pos position: {first_pos}")
                return []
            if last_pos[0] < 0 or last_pos[0] >= world_width or last_pos[1] < 0 or last_pos[1] >= world_height:
                print(f"Invalid last_pos position: {last_pos}")
                return []
            
            # Check if first_pos or last_pos is a wall
            if surface.get_at(first_pos) != WHITE:
                print(f"first_pos position is a wall: {first_pos}")
                return []
            if surface.get_at(last_pos) != WHITE:
                print(f"last_pos position is a wall: {last_pos}")
                return []
        except IndexError:
            print(f"Index error checking first_pos/last_pos positions: {first_pos}, {last_pos}")
            return []
        
        # If first_pos and last_pos are the same, return a single-point path
        if first_pos == last_pos:
            return [first_pos]
        
        
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),(1, 1), (1, -1), (-1, 1), (-1, -1)  
        ]
        
        # Create seen set to keep track of evaluated nodes
        seen = set()
        
        # Format: (distance, count, node)
        pq = []
        count = 0  # Tiebreaker for nodes with same distance
        heapq.heappush(pq, (0, count, first_pos))
        
        # Dictionary to store distances from first_pos to each node
        distances = {first_pos: 0}
        
        # Dictionary to store the parent of each node (for path reconstruction)
        came_from = {}
        
        # Set a maximum number of its to prevent infinite loops
        max_its = world_width * world_height
        its = 0
        
        while pq and its < max_its:
            its += 1
            
            # Get node with smallest distance from first_pos
            current_dist, _, current = heapq.heappop(pq)
            
            # Skip if already seen
            if current in seen:
                continue
                
            # Mark as seen
            seen.add(current)
            
            # Check if we've reached the last_pos
            if current[0] == last_pos[0] and current[1] == last_pos[1]:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(first_pos)
                path.reverse()
                return path
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if (neighbor[0] < 0 or neighbor[0] >= world_width or
                    neighbor[1] < 0 or neighbor[1] >= world_height):
                    continue
                
                # Skip if already seen
                if neighbor in seen:
                    continue
                
                # Skip if neighbor is not white or too close to a wall
                # (except for last_pos point which is allowed to be near walls)
                if neighbor != last_pos:
                    if not self.not_very_close(surface, neighbor, world_width, world_height, buffer_zone):
                        continue
                else:
                    # For the last_pos, just make sure it's not a wall
                    try:
                        if surface.get_at((int(neighbor[0]), int(neighbor[1]))) != WHITE:
                            continue
                    except IndexError:
                        continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                new_distance = distances[current] + movement_cost
                
                # Update distance if this path is better
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    came_from[neighbor] = current
                    
                    # Add to priority queue
                    count += 1
                    heapq.heappush(pq, (new_distance, count, neighbor))
    def get_meeting_point(self, surface, world_height, world_width, pos1, pos2):

            SAFE_COLOR = (255, 255, 255)
            safety_margin = 5
            
            # First try the exact midpoint
            x_mid = (pos1[0] + pos2[0]) / 2.0
            y_mid = (pos1[1] + pos2[1]) / 2.0
            proposed_midpoint = (int(x_mid), int(y_mid))
            
            # Check if midpoint is valid and safe distance from walls
            if self.not_very_close(surface, proposed_midpoint, world_width, world_height, safety_margin):
                return proposed_midpoint
            
            # position of agent 1
            return pos1
