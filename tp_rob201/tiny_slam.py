""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.counter = 0
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        max_Range = lidar.max_range
        distance = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        
        angles_filtered = angles[distance<max_Range]
        distances_filtered = distance[distance<max_Range]
        x, y = self.pol_to_coord(pose, distances_filtered, angles_filtered)
        
        idx_grid = self.grid.conv_world_to_map(x, y)
        #Removing points out of map and values = 0 (log(0) = -inf)
        idx_grid = (
                np.clip(idx_grid[0], 0, self.grid.occupancy_map.shape[0] - 1),
                np.clip(idx_grid[1], 0, self.grid.occupancy_map.shape[1] - 1)
                )
        score = np.sum(self.grid.occupancy_map[idx_grid])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            return self.odom_pose_ref
        # Odom
        odom_pose = odom_pose + odom_pose_ref
        xref = odom_pose[0]
        yref = odom_pose[1]
        # Absolut
        alpha0 = np.arctan2(yref, xref)
        d0 = np.sqrt(xref**2 + yref**2)
        x = xref + d0 * np.cos(odom_pose[2] + alpha0)
        y = xref + d0 * np.cos(odom_pose[2] + alpha0)
        corrected_pose = [x, y, alpha0]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        odom_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar = lidar, pose =odom_pose)
        N = 10
        for i in range(N):
            sigma = 0.5
            offset = np.random.normal(0, sigma, size=3) # e.g., array([0.12, -0.43, 0.05])
            odom_pose_ref_offset = self.odom_pose_ref + offset
            new_pose = self.get_corrected_pose(raw_odom_pose, odom_pose_ref_offset)
            test_score = self._score(lidar = lidar, pose = new_pose)
            if test_score > best_score:
                best_score = test_score
                self.odom_pose_ref = odom_pose_ref_offset

        return best_score

    def pol_to_coord(self, pose, dists, angles):
        """
        Convert LiDAR ranges and angles to (x, y) world coordinates.
        dists and angles should be lidar arrays.
        """
        angles_world = pose[2] + angles  # rotate by robot orientation
        x = pose[0] + dists * np.cos(angles_world)
        y = pose[1] + dists * np.sin(angles_world)
        return x, y
        
    def update_map(self, lidar, pose, goal=None):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        x, y = self.pol_to_coord(pose,lidar.get_sensor_values(), lidar.get_ray_angles())
        
        for x_coord, y_coord in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], x_coord, y_coord, -1.0)
        self.grid.add_map_points(x, y, 2.0)
        
        self.counter += 1
        if self.counter == 10:
            self.grid.display_cv(robot_pose = pose, goal=goal)
            self.counter = 0

    # def compute(self):
    #     """ Useless function, just for the exercise on using the profiler """
    #     # Remove after TP1

    #     ranges = np.random.rand(3600)
    #     ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

    #     # Poor implementation of polar to cartesian conversion
    #     points = []
    #     for i in range(3600):
    #         pt_x = ranges[i] * np.cos(ray_angles[i])
    #         pt_y = ranges[i] * np.sin(ray_angles[i])
    #         points.append([pt_x, pt_y])
