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
    speed = 0.3  # Normal forward speed
    rotation_speed = 0.0  # No rotation by default

    # Obstacle detection: Check distance in front
    front_idx = len(laser_dist) // 2  # Safer way to access the front distance
    min_abs_dist = min(laser_dist)
    min_dist = min(laser_dist[front_idx - 10:front_idx + 10])  # Check a range around the front
    if min_dist < 60.0:  # Obstacle detected in front
        speed = 0.0  # Slow down
        # Obstacle on the left, turn right
        if(sum(laser_dist[:front_idx]) < sum(laser_dist[front_idx+2:])):
            rotation_speed = 0.15
        else:
            rotation_speed = -0.15
    else:
        if(laser_dist[90] < min_abs_dist + 10):  # Obstacle on the left
            speed = 0.2
            rotation_speed = 0.15
        elif(laser_dist[270] < min_abs_dist + 10):  # Obstacle on the left
            speed = 0.2
            rotation_speed = -0.15
    
    # Return movement command
    command = {
        "forward": speed,
        "rotation": rotation_speed
    }
    return command

def attractive_field(Kv, gx, gy, xp, yp):
    """
    Compute attractive field for goal reaching
    Kv : attractive gain
    goal_pose : [gx, gy]  target pose in odom or world frame
    current_pose : [xp, yp]  current pose in odom or world frame
    """
    dRobo_Goal = np.hypot(gx - xp, gy - yp)
    dist = np.array([gx - xp, gy - yp])

    if dRobo_Goal > 1e-6:  # Avoid division by zero
        gradient_f = (Kv / dRobo_Goal) * dist
    else:
        gradient_f = np.zeros(2)

    return gradient_f, dRobo_Goal

def repulsive_field(lidar: 'LidarSensor', repulsive_gain,
                   SAFE_DIST, current_pose):
    """
    Compute repulsive field vector for obstacle avoidance.
    
    Args:
        lidar: Lidar sensor object with get_sensor_values() and get_ray_angles() methods
        repulsive_gain: Scaling factor for repulsive force (Kobs)
        safe_distance: Maximum influence distance for obstacles (meters)
        current_pose: Current robot pose [x, y, theta] in world coordinates
        
    Returns:
        np.ndarray: 2D repulsive gradient vector [Fx, Fy]
    """
    # Get filtered lidar measurements within safe distance
    distances = np.array(lidar.get_sensor_values())
    angles = np.array(lidar.get_ray_angles())
    
    # Filter out measurements beyond safe distance
    valid_mask = distances < SAFE_DIST
    distances = distances[valid_mask]
    angles = angles[valid_mask]
    
    # Return zero vector if no obstacles in range
    if len(distances) == 0:
        return np.zeros(2)
    
    # Find closest obstacle
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    min_angle = angles[min_idx]
    
    # Calculate obstacle position in world frame
    x_robot, y_robot, theta_robot = current_pose
    obstacle_x = x_robot + min_dist * np.cos(min_angle + theta_robot)
    obstacle_y = y_robot + min_dist * np.sin(min_angle + theta_robot)
    obstacle_pos = np.array([obstacle_x, obstacle_y])
    
    # Calculate robot-to-obstacle vector and distance
    robot_pos = current_pose[:2]
    robot_to_obstacle = obstacle_pos - robot_pos
    distance = np.linalg.norm(robot_to_obstacle)
    
    # Compute repulsive gradient
    if distance <= SAFE_DIST and distance > 1e-6:  # Avoid division by zero
        scale_factor = (1.0/distance - 1.0/SAFE_DIST)
        gradient = (repulsive_gain / (distance**3)) * scale_factor * robot_to_obstacle
    else:
        gradient = np.zeros(2)
    print(min_dist)
    return gradient

def space_field(lidar: 'LidarSensor', Kv, current_pose):
    """
    Compute repulsive field vector for obstacle avoidance.
    
    Args:
        lidar: Lidar sensor object with get_sensor_values() and get_ray_angles() methods
        repulsive_gain: Scaling factor for repulsive force (Kobs)
        safe_distance: Maximum influence distance for obstacles (meters)
        current_pose: Current robot pose [x, y, theta] in world coordinates
        
    Returns:
        np.ndarray: 2D repulsive gradient vector [Fx, Fy]
    """
    # Get filtered lidar measurements within safe distance
    distances = np.array(lidar.get_sensor_values())
    angles = np.array(lidar.get_ray_angles())
    max_range = lidar.max_range

    # Filter out invalid distance readings
    valid_mask = distances < max_range

    # Build index mask: keep indexes in [90, 270]
    n = len(distances)  # Total number of LiDAR rays
    index_mask = np.zeros(n, dtype=bool)
    index_mask[270:360] = True       # [90, 270] inclusive

    # Combine with valid distance readings
    final_mask = valid_mask & index_mask

    # Apply mask
    distances = distances[final_mask]
    angles = angles[final_mask]
    # Return zero vector if no obstacles in range
    if len(distances) == 0:
        return np.zeros(2)
    
    # Find  obstacle
    max_idx = np.argmax(distances)
    max_dists = distances[max_idx]
    max_angles = angles[max_idx]
    gradient = np.zeros(2)
    # Calculate obstacle position in world frame
    for max_dist, max_angle in zip(max_dists, max_angles):
        x_robot, y_robot, theta_robot = current_pose
        obstacle_x = x_robot + max_dist * np.cos(max_angle + theta_robot)
        obstacle_y = y_robot + max_dist * np.sin(max_angle + theta_robot)
        obstacle_pos = np.array([obstacle_x, obstacle_y])
        
        # Calculate robot-to-obstacle vector and distance
        robot_pos = current_pose[:2]
        robot_to_obstacle = obstacle_pos - robot_pos
        dRobo_Space = np.linalg.norm(robot_to_obstacle)  

        if distance <= SAFE_DIST and distance > 1e-6:  # Avoid division by zero
            scale_factor = (1.0/distance - 1.0/SAFE_DIST)
            gradient += (repulsive_gain / (distance**3)) * scale_factor * robot_to_obstacle
        else:
            gradient += np.zeros(2)

    return gradient_s

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
    Kv = 0.20  # Attractive gain
    Kw = 1 # Angular gain
    Kobs = 6000 # Repulsive gain
    SAFE_DIST = 40.0  # Obstacle influence distance (meters)
    phi_max = 0.030  # Maximum angle for full speed
    max_rot_speed = 1.0
    min_dist_threshold = 5.0  # Minimum distance to consider goal reached
    
    
    # Initialize commands
    vitesse = 0.0
    w_speed = 0.0
    
    gradient_f, dRobo_Goal = attractive_field(Kv, goal_pose[0], goal_pose[1], current_pose[0], current_pose[1])
    goal_reachead = False
    if dRobo_Goal < min_dist_threshold:
        print("Goal reached")
        goal_reachead = True
        return {"forward": 0.0, "rotation": 0.0}, goal_reachead
    

    gradient_r = repulsive_field(lidar, Kobs, SAFE_DIST, current_pose)
    F_total = gradient_f  - gradient_r

    # Calculate desired movement direction
    desired_angle = np.arctan2(F_total[1], F_total[0])
    theta = current_pose[2]
    phi_R = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
    #phi_R = desired_angle - theta
    
    # Calculate forward speed based on alignment
    module_gradient_F = np.linalg.norm(F_total)
    if abs(phi_R) < phi_max:
        vitesse = Kv * module_gradient_F
    else:
        vitesse = Kv * module_gradient_F * (phi_max / abs(phi_R))
    
    vitesse = np.clip(vitesse,-max_rot_speed,max_rot_speed)
    w_speed = np.clip(Kw * phi_R, -max_rot_speed, max_rot_speed)
    return {"forward": vitesse, "rotation": w_speed}, goal_reachead