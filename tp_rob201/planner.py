"""
Planner class
Implementation of A*
"""

import numpy as np
from collections import defaultdict
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
    def get_neighbors(self, current_cell):
        """
        Get the neighbors of a cell in the occupancy grid
        current_cell : [x, y] nparray, current cell in map coordinates
        """
        # TODO for TP5
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor = current_cell + np.array([i, j])
                # Check if the neighbor is within the grid bounds
                # if (0 <= neighbor[0] < self.grid.occupancy_map.shape[0] and
                #         0 <= neighbor[1] < self.grid.occupancy_map.shape[1]):
                # not necessary, the walls are big enough
                neighbors.append(neighbor)
        return neighbors
    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        returns a path in world coordinates
        """
        # TODO for TP5

        path = self.A_Star(start, goal)
        for i in range(len(path)):
            xp, yp = path[i]
            path[i] = self.grid.conv_map_to_world(xp, yp)
        return path
    
    def heuristic(self, cell1, cell2):
        """
        Heuristic function for A* algorithm
        cell1 : [x, y] nparray, first cell in map coordinates
        cell2 : [x, y] nparray, second cell in map coordinates

        Returns the Euclidean distance between the two cells
        """
        # TODO for TP5
        return np.linalg.norm(cell1 - cell2)	

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
    

    def reconstruct_path(self, cameFrom, current):
        total_path = [current]
        while tuple(current) in cameFrom:
            current = cameFrom[tuple(current)]
            total_path.insert(0, current)
        return total_path

    def A_Star(self, start, goal):
        xs_map, ys_map = self.grid.conv_world_to_map(start[0], start[1])
        xg_map, yg_map = self.grid.conv_world_to_map(goal[0], goal[1])
        start = [xs_map, ys_map]
        goal = [xg_map, yg_map]
        # The set of discovered nodes that may need to be (re -) expanded.
        # Initially , only the start node is known.
        # This is usually implemented as a min -heap or priority queue rather than a
        # hash -set.
        openSet = {tuple(start)}
        # For node n, cameFrom[n] is the node immediately preceding it on the cheapest
        # path from the start to n currently known.
        cameFrom = {}
        # For node n, gScore[n] is the cost of the cheapest path from start to n
        # currently known.
        gScore = defaultdict(lambda: np.inf)
        fScore = defaultdict(lambda: np.inf)
        # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current
        # best guess as to
        # how cheap a path could be from start to finish if it goes through n.
        gScore[tuple(start)] = 0
        fScore[tuple(start)] = self.heuristic(np.array(start), np.array(goal))

        while openSet:
            # This operation can occur in O(Log(N)) time if openSet is a min -heap or a
            # priority queue
            current = min(openSet, key=lambda n: fScore[n])
            if current == tuple(goal):
                return self.reconstruct_path(cameFrom, current)
            
            openSet.remove(current)
            neighbors = self.get_neighbors(np.array(current))
            for voisin in neighbors:
                # d(current ,neighbor) is the weight of the edge from current to neighbor
                # tentative_gScore is the distance from start to the neighbor through current
                voisin_tuple = tuple(voisin)
                tentative_gScore = gScore[current] + self.heuristic(np.array(current), voisin)
                if tentative_gScore < gScore[voisin_tuple]:
                    # This path to neighbor is better than any previous one. Record it
                    cameFrom[voisin_tuple] = current
                    gScore[voisin_tuple] = tentative_gScore
                    fScore[voisin_tuple] = tentative_gScore + self.heuristic(voisin, np.array(goal))
                    if voisin_tuple not in openSet:
                        openSet.add(voisin_tuple)
        return []
