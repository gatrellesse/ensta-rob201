"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams
import time 
import copy
from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, wall_Follower
from occupancy_grid import OccupancyGrid
from planner import Planner
from collections import deque



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
        self.traj_goals = [[-490,0], [-810, -140], [-900, -300]]
        #self.traj_goals = [[-490,0],[-490,-60]]
        self.traj_original = None
        #self.traj_goals = [[-70,0]]
        #self.traj_goals =[[-310,20]]
        self.goal = self.traj_goals.pop(0)
        self.goal_reachead = False
        self.seuil = 200
        self.last_wall_side = "unknown"
        self.control_mode = "Normal" # Normal, Wall, Planner
        self.wall_counter = 0
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
        self.occupancy_grid_Fat = None
        self.tiny_slam = TinySlam(self.occupancy_grid)

        # storage for pose after localization
        self.stored_poses = deque(maxlen=300)
        self.corrected_pose = np.array([0, 0, 0])
        
    
    def control(self):
        """
        Main control function executed at each time step
        """
        #TD3
        #self.tiny_slam.update_map(lidar = self.lidar(),pose = self.odometer_values(), goal=self.goal)
        
        #TD4
        best_score = self.tiny_slam.localise(self.lidar(), self.odometer_values())            
        
        # We update only if the odotometry is not too bad(test the seuil and
        # see if the map doesnt change)
        if((self.control_mode != "Wall" and best_score > self.seuil) or self.counter <= 20):
            #print(f"Score: {best_score}")
            self.counter += 1
            if(self.counter == 20):
                self.seuil = 5000
            
            self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values(), self.tiny_slam.odom_pose_ref)
            self.tiny_slam.update_map(lidar = self.lidar(), pose= self.corrected_pose, goal=self.goal, traj=self.traj_original, mode = self.control_mode)

        # To run with wall follower as a debugger when crashing just uncomment the line below
        # i didnt check if it works 100%
        # if(self.is_stuck() and self.control_mode != "Planner"): # Planner can`t be stuck
        #     self.control_mode = "Wall"
        #     self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        #     self.tiny_slam.update_map(lidar = self.lidar(), pose= self.corrected_pose, goal=self.goal)
        #     return self.control_wall_Follower()
        
        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command
    
    def is_stuck(self, std_thresh=15.0):
        self.stored_poses.append(self.corrected_pose[:2])
        if len(self.stored_poses) < 300:
            return False
        
        x_coords, y_coords = zip(*self.stored_poses)
        std_x = np.std(x_coords)
        std_y = np.std(y_coords)
        # print("Std x: ",std_x," Std y: ",std_y)
        # If std deviation in x and y are both small, it's not moving much
        if std_x < std_thresh and std_y < std_thresh:  # e.g., 5 cm
            return True
    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.corrected_pose
        # Compute new command speed to perform obstacle avoidance
        if self.goal_reachead == False:
            command, self.goal_reachead = potential_field_control(self.lidar(), pose, self.goal)
        else:
            command = {"forward": 0.0, "rotation": 0.0}
            if(self.traj_goals):
                self.goal = self.traj_goals.pop(0)
                self.goal_reachead = False
                return command
            elif(self.control_mode != "Planner"):#Starts Planner with fat map
                print("Planner has started, returning to origin.")
                # self.occupancy_grid_Fat = copy.deepcopy(self.occupancy_grid)
                cv_out_temp = self.occupancy_grid.cv_out
                self.occupancy_grid.cv_out = None
                self.occupancy_grid_Fat = copy.deepcopy(self.occupancy_grid)
                self.occupancy_grid.cv_out = cv_out_temp

                self.occupancy_grid_Fat.enlarge_obstacles(spread_distance = 11)
                self.planner = Planner(self.occupancy_grid_Fat)
                self.traj_goals = self.planner.plan(np.array([0, 0, 0]), self.goal)
                self.traj_original = copy.deepcopy(self.traj_goals)
                self.goal = self.traj_goals.pop(0)
                self.goal_reachead = False
                self.control_mode = "Planner"
                self.seuil = 0
            else:
                print("Final Goal reached, stopping")
                self.traj_original = None
                command = {"forward": 0.0, "rotation": 0.0}
                return command

        return command

    def control_wall_Follower(self):
        
        if self.goal_reachead == False:
            print("Wall following mode activated with ", self.last_wall_side)
            self.wall_counter += 1
            command, self.last_wall_side , mode_flag = wall_Follower(self.lidar(), self.last_wall_side, self.wall_counter)
            if self.control_mode == "Wall" and mode_flag == False:
                print("Turning off wall mode and reseting stored poses")
                self.wall_counter = 0
                self.control_mode = "Normal" 
                self.stored_poses.clear()
            return command