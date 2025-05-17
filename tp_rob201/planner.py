"""
Planner class
Implementation of A* algorithm with optimizations
"""

import numpy as np
from collections import defaultdict
import heapq
from occupancy_grid import OccupancyGrid


class Planner:
    """Optimized occupancy grid Planner using A* algorithm"""

    # Constants for occupancy values
    FREE = 0
    OCCUPIED = 20

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.odom_pose_ref = np.array([0, 0, 0])  # Origin of odom frame in map frame

    def is_valid_cell(self, cell: np.ndarray) -> bool:
        """Check if cell is within bounds and not occupied"""
        x, y = cell
        return (0 <= x < self.grid.x_max_map and 
                0 <= y < self.grid.y_max_map and
                self.grid.occupancy_map[(x, y)] < -self.OCCUPIED)

    def get_neighbors(self, current_cell: np.ndarray) -> list:
        """
        Get valid neighbors of a cell in the occupancy grid
        Args:
            current_cell: [x, y] array, current cell in map coordinates
        Returns:
            List of valid neighbor coordinates
        """
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip current cell
                neighbor = current_cell + np.array([i, j])
                if self.is_valid_cell(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def get_weight(self, nodeA: np.ndarray, nodeB: np.ndarray) -> float:
        """
        Returns movement cost between adjacent nodes
        Args:
            nodeA, nodeB: Adjacent nodes in map coordinates
        Returns:
            1.0 for cardinal moves, 1.4 for diagonal moves
        """
        dx, dy = abs(nodeB - nodeA)
        return 1.4 if dx and dy else 1.0

    def heuristic(self, cell1: np.ndarray, cell2: np.ndarray) -> float:
        """
        Euclidean distance heuristic for A*
        Args:
            cell1, cell2: [x, y] arrays in map coordinates
        Returns:
            Euclidean distance between cells
        """
        return np.linalg.norm(cell1 - cell2)

    def reconstruct_path(self, cameFrom: dict, current: tuple) -> list:
        """
        Reconstruct path from cameFrom dictionary
        Args:
            cameFrom: Dictionary of node predecessors
            current: Goal node to start reconstruction from
        Returns:
            List of nodes from start to goal
        """
        total_path = [current]
        while current in cameFrom:
            current = cameFrom[current]
            total_path.append(current)
        return total_path # goal --> start

    def A_Star(self, start: list, goal: list) -> list:
        """
        A* algorithm implementation
        Args:
            start: [x, y] start position in world coordinates
            goal: [x, y] goal position in world coordinates
        Returns:
            Path in map coordinates or empty list if no path found
        """
        # Convert to map coordinates
        xs, ys = self.grid.conv_world_to_map(start[0], start[1])
        xg, yg = self.grid.conv_world_to_map(goal[0], goal[1])
        start = (xs, ys)
        goal = (xg, yg)

        # Priority queue for open set
        openSet = []
        heapq.heappush(openSet, (0, start))

        # Dictionaries for path reconstruction and scoring
        cameFrom = {}
        gScore = defaultdict(lambda: np.inf)
        gScore[start] = 0
        fScore = defaultdict(lambda: np.inf)
        fScore[start] = self.heuristic(np.array(start), np.array(goal))

        while openSet:
            _, current = heapq.heappop(openSet)

            if current == goal:
                return self.reconstruct_path(cameFrom, current)

            for neighbor in self.get_neighbors(np.array(current)):
                neighbor = tuple(neighbor)
                tentative_gScore = gScore[current] + self.get_weight(
                    np.array(current), np.array(neighbor))

                if tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + self.heuristic(
                        np.array(neighbor), np.array(goal))
                    heapq.heappush(openSet, (fScore[neighbor], neighbor))

        return []  # No path found

    def plan(self, start: np.ndarray, goal: np.ndarray) -> list:
        """
        Compute path using A* and convert to world coordinates
        Args:
            start: [x, y, theta] start pose in world coordinates
            goal: [x, y, theta] goal pose in world coordinates
        Returns:
            Path in world coordinates
        """
        path = self.A_Star(start, goal)
        return [self.grid.conv_map_to_world(x, y) for x, y in path]

    def simplify_path(self, path: list) -> list:
        return path
    
    def explore_frontiers(self) -> np.ndarray:
        """Frontier-based exploration placeholder"""
        return np.array([0, 0, 0])