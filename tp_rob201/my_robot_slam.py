"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        self.goal = None
        self.goal_reachead = False
        self.started = False
        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        

    def control(self):
        """
        Main control function executed at each time step
        """
        #TD3
        #self.tiny_slam.update_map(lidar = self.lidar(),pose = self.odometer_values(), goal=self.goal)

        #TD4
        seuil = 0
        
        best_score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
        self.tiny_slam.update_map(lidar = self.lidar(),pose= self.odometer_values(), goal=self.goal)
        print(f"Score: {best_score}, Pose: {self.corrected_pose}")

        if best_score > seuil:
            print(f"Score: {best_score}, Pose: {self.corrected_pose}")
            self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values(), self.tiny_slam.odom_pose_ref)
            return self.control_tp2()
        return {"forward": 0.0, "rotation": 0.0}  

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        # goal_pose : [x, y, theta] nparray, target pose in odom or world frame
        self.goal = [-490,-400]
        
        # Compute new command speed to perform obstacle avoidance
        if self.goal_reachead == False:
            command, self.goal_reachead = potential_field_control(self.lidar(), pose, self.goal)
        else:
            command = {"forward": 0.0, "rotation": 0.0}  
        return command
