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
    # Parameters
    Kv = 0.4  # Attractive gain
    Kw = 0.5  # Angular gain
    Kobs = 0.4  # Repulsive gain
    SAFE_DIST = 10.0  # Obstacle influence distance (meters)
    phi_max = 3.0  # Maximum angle for full speed
    max_rot_speed = 1.0
    min_dist_threshold = 0.5  # Minimum distance to consider goal reached
    
    
    # Initialize commands
    vitesse = 0.0
    w_speed = 0.0
    
    # Calculate attractive force (goal gradient)
    dx = goal_pose[0] - current_pose[0]
    dy = goal_pose[1] - current_pose[1]
    dRobo_Goal = np.hypot(dx, dy)
    
    # Normalize and apply gain
    if dRobo_Goal > 1e-5:  # Avoid division by zero
        gradient_f = np.array([Kv * dx/dRobo_Goal, Kv * dy/dRobo_Goal])
    else:
        gradient_f = np.zeros(2)
    
    # Check if goal is reached
    goal_reachead = False
    if dRobo_Goal < min_dist_threshold:
        print("Goal reached")
        goal_reachead = True
        return {"forward": 0.0, "rotation": 0.0}
    
    # Initialize repulsive force
    gradient_r = np.zeros(2)
    
    # Get lidar data
    laser_dist = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()
    
    # Calculate repulsive forces from all relevant obstacles
    for dist, angle in zip(laser_dist, ray_angles):
          if dist < SAFE_DIST:
            magnitude = (Kobs / (dist**3)) * (1.0/dist - 1.0/SAFE_DIST) 
            if magnitude > 0:
                # Force direction is away from the obstacle
                gradient_r[0] -= magnitude * np.cos(angle)
                gradient_r[1] -= magnitude * np.sin(angle)
    
    # Combine forces
    F_total = gradient_f + gradient_r
    
    # Calculate desired movement direction
    desired_angle = np.arctan2(F_total[1], F_total[0])
    theta = current_pose[2]
    phi_R = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
    #phi_R = desired_angle - theta
    # Calculate rotation speed
    w_speed = np.clip(Kw * phi_R, -max_rot_speed, max_rot_speed)
    # Calculate forward speed based on alignment
    module_gradient_f = np.linalg.norm(gradient_f)
    if abs(phi_R) < phi_max:
        vitesse = Kv * module_gradient_f
    else:
        vitesse = Kv * module_gradient_f * (phi_max / abs(phi_R))
    
    # Ensure we don't move backward
    vitesse = max(0, vitesse)
    
    return {"forward": vitesse, "rotation": w_speed}, goal_reachead