import pygame
import heapq
import math

class Planner:
    def __init__(self):
        self.path = None

    def heuristic(self, point1, point2):
        """Heuristic function using Euclidean distance."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_neighbors(self, point, surface, world_height, world_width):
        """Generates valid neighboring points for pathfinding."""
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = int(point[0] + dx), int(point[1] + dy)
            if 0 <= nx < world_width and 0 <= ny < world_height:
                if surface.get_at((nx, ny))[:3] == (255, 255, 255):  # White areas are navigable
                    neighbors.append((nx, ny))
        return neighbors

    def a_star(self, surface, world_height, world_width, start, goal):
        """A* Algorithm for optimal pathfinding."""
        start = tuple(start)
        goal = tuple(goal)
        open_list = []
        heapq.heappush(open_list, (0, start))  # (cost, point)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current, surface, world_height, world_width):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []  # Return empty list if no path is found

    def reconstruct_path(self, came_from, current):
        """Backtrack to reconstruct the optimal path."""
        self.path = [current]
        while current in came_from:
            current = came_from[current]
            self.path.append(current)
        self.path.reverse()
        return self.path

    
    def get_path(self, surface, world_height, world_width, start, goal):
        """Combines pathfinding with path smoothing logic."""
        self.path = self.a_star(surface, world_height, world_width, start, goal)
        print(self.path)
        return self.path
