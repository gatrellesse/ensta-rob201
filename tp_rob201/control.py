""" A set of robotics control functions """

import random
import numpy as np
from math import sqrt


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance using LiDAR.
    
    Parameters:
    lidar : Placebot object with lidar data.
    
    Returns:
    dict : Movement command with "forward" and "rotation".
    """
    # Get LiDAR data
    laser_dist = lidar.get_sensor_values()  # List of distances
    laser_angle = lidar.get_ray_angles()  # Corresponding angles

    # Convert polar coordinates to Cartesian (x, y)
    # x0 = laser_dist * np.cos(laser_angle)
    # y0 = laser_dist * np.sin(laser_angle)

    # Default movement parameters
    speed = 0.0  # Normal forward speed
    rotation_speed = 0.0  # No rotation by default

    # Obstacle detection: Check distance in front
    front_idx = len(laser_dist) // 2  # Safer way to access the front distance
    min_abs_dist = min(laser_dist)
    min_dist = min(laser_dist[front_idx - 10:front_idx + 10])  # Check a range around the front
    if min_dist < 40.0:  # Obstacle detected in front
        speed = 0.0  # Slow down
        # Obstacle on the left, turn right
        if(sum(laser_dist[:front_idx]) < sum(laser_dist[front_idx+2:])):
            rotation_speed = 0.4
        else:
            rotation_speed = -0.4
    else:
        if(laser_dist[90] < min_abs_dist + 10):  # Obstacle on the left
            speed = 0.2
            rotation_speed = 0.4
        elif(laser_dist[270] < min_abs_dist + 10):  # Obstacle on the left
            speed = 0.2
            rotation_speed = -0.4
    
    # Return movement command
    command = {
        "forward": speed,
        "rotation": rotation_speed
    }
    print(command)
    return command




def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : -[x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    
    Kv = 0.2
    Kw = 2.0

    dRobo_Goal = sqrt((goal_pose[0] - current_pose[0])**2 + (goal_pose[1] - current_pose[1])**2)
    gradient_f = Kv * (goal_pose - current_pose)/dRobo_Goal
    module_gradient_f = sqrt(gradient_f[0]**2 + gradient_f[1]**2)
    phi_max = np.pi/2

    laser_dist = lidar.get_sensor_values()  # List of distances
    # Parameters
    SAFE_DIST = 10.0  # Ignore obstacles beyond this distance (meters)

    # Initialize with infinity (to handle any LiDAR range)
    min1, min2 = float('inf'), float('inf')
    idx1, idx2 = -1, -1

    # Find the two closest obstacles
    for i, dist in enumerate(laser_dist):
        if dist > SAFE_DIST:  # Skip if too far
            continue
        if dist < min1:
            min2, idx2 = min1, idx1  # Shift previous closest to second closest
            min1, idx1 = dist, i     
        elif dist < min2:
            min2, idx2 = dist, i    

    obstacles = [] 
    if idx1 != -1:  # If at least one obstacle found
        obs_vector = np.zeros(3)  
        angle = lidar.get_ray_angles()[idx1]  
        obs_vector[0] = np.cos(angle) / (min1 + 1e-5)  # Avoid division by zero
        obs_vector[1] = np.sin(angle) / (min1 + 1e-5)  # Inverse distance scaling
        obs_vector[2] = angle
        obstacles.append(obs_vector)

    # Optional: Combine both closest obstacles
    if idx2 != -1:  # If a second obstacle exists
        obs_vector = np.zeros(3)    
        angle = lidar.get_ray_angles()[idx2]
        obs_vector[0] += np.cos(angle) / (min2 + 1e-5)
        obs_vector[1] += np.sin(angle) / (min2 + 1e-5)
        obs_vector[2] = angle
        obstacles.append(obs_vector)
    Kobs = 0.8 # K repulsive
    F_total = 0
    for obs in obstacles:
        gradient_repulsive =(Kobs / (min1 **3))  * (1/min1 -1/SAFE_DIST) * (obs_vector - current_pose)
        F_total += gradient_repulsive

    # Angle entre la direction du robot et le gradient
    F_total += gradient_f
    desired_angle = np.arctan2(F_total[1], F_total[0])
    theta = current_pose[2]
    phi_R = np.arctan2(np.sin(desired_angle - theta), 
                               np.cos(desired_angle - theta))
    #phi_r = desired_angle - theta
    
    max_rot_speed = 0.5
    w_speed = np.clip(Kw * phi_R, -max_rot_speed, max_rot_speed)
    if abs(phi_R) < phi_max:
        vitesse = Kv * module_gradient_f
    else:
        vitesse = Kv * module_gradient_f * (phi_max / abs(phi_R))
    
    distance_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])
    # Check if gradient is negligible OR robot is very close to goal
    if (abs(gradient_f[0]) < 0.05 and abs(gradient_f[1]) < 0.05) \
    or (distance_to_goal < 4):
        vitesse = 0.0
        w_speed = 0.0

    command = {"forward": vitesse,
               "rotation": w_speed}
    return command
