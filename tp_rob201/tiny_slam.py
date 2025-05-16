""" A simple robotics navigation code including SLAM, exploration, planning"""
import time
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
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        
        angles_filtered = angles[distances<max_Range]
        distances_filtered = distances[distances<max_Range]
        x_world, y_world = self.pol_to_coord(pose, distances_filtered, angles_filtered)
        
        idx_grid = self.grid.conv_world_to_map(x_world, y_world)
        #Removing points out of map and values = 0 (log(0) = -inf)
        idx_grid = (
                np.clip(idx_grid[0], 0, self.grid.occupancy_map.shape[0] - 1),
                np.clip(idx_grid[1], 0, self.grid.occupancy_map.shape[1] - 1)
                )
        score = np.sum(self.grid.occupancy_map[idx_grid])
        #max score:+360*40 = 14400 --> All points are obstacles
        #min score:+360*40 =-14400 --> None of the points are obstacles
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
            odom_pose_ref = self.odom_pose_ref
        # Robot Odom
        x0, y0, theta0 = odom_pose
        x0ref, y0ref, theta0ref = odom_pose_ref
        # Robot Absolut
        alpha0 = np.arctan2(y0, x0)
        d0 = np.sqrt(x0**2 + y0**2)

        x = x0ref + d0 * np.cos(theta0ref + alpha0)
        y = y0ref + d0 * np.sin(theta0ref + alpha0)
        theta = theta0ref + theta0
        corrected_pose = np.array([x, y, theta])
        
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
        current_odom_pose_ref = self.odom_pose_ref
        N = 100
        for i in range(N):
            sigma = 0.10
            offset = np.random.normal(0, sigma, size=2) # e.g., array([0.12, -0.43, 0.05])
            sigmaTheta = 0.05
            offsetTheta = np.random.normal(0, sigmaTheta, size=1)
            offset = np.append(offset, offsetTheta)

            odom_pose_ref_offset = current_odom_pose_ref + offset
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
        x, y = self.pol_to_coord(pose, lidar.get_sensor_values(), lidar.get_ray_angles())
        for x_coord, y_coord in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], x_coord, y_coord, -1.0)
        self.grid.add_map_points(x, y, 10.0)
        self.counter += 1
        if self.counter == 10:
            self.grid.display_cv(robot_pose = pose, goal=goal)
            self.counter = 0
