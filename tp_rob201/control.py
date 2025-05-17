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

def repulsive_field(lidar, repulsive_gain,
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
    return gradient

def wall_Follower(lidar, last_side, counter ,clearance_wall=25, dst_lim = 60):
    """
    Wall following using LiDAR. Follows a wall on the specified side.
    
    Parameters:
    - lidar: LiDAR sensor with get_sensor_values()
    - clearance_wall: Desired distance from the wall (in cm)
    - side: "right" or "left"
    
    Down idx = 0
    Right idx = 90
    Front idx = 180
    Left idx = 270
    ref: https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html
    Returns:
    - dict with "forward" and "rotation" keys
    """
    
    distances = np.array(lidar.get_sensor_values())

    # Parameters
    forward = 0.02
    Kp_dist = 0.01   # Gain to correct distance error
    Kp_angle = 0.02     # Gain to align with wall
    min_idx = np.argmin(distances)
    
    side = "left"
    """
             180
              y+
              ↑
    left      |    right
              |
    y– ←----- 0 -----→ x+ 90
        (robot)
    """
    if min_idx < 180:
        side = "right"

    if side == "right":
        idx_a = 90  # Right
        idx_b = 150  # Front-right
    else:
        idx_a = 270  # Left
        idx_b = 240  # Front-left
    
    #
    # Get distances at two angles
    dist_a = distances[idx_a]
    dist_b = distances[idx_b]

    # Compute angle of the wall (relative to robot)
    alpha = np.arctan2(dist_b * np.cos(np.radians(idx_b - idx_a)) - dist_a,
                       dist_b * np.sin(np.radians(idx_b - idx_a)))
    
    # Estimate distance from wall (perpendicular)
    distance_to_wall = dist_a * np.cos(alpha)

    # Compute errors
    error_dist = clearance_wall - distance_to_wall
    error_angle = alpha

    # Control rotation
    rotation = Kp_dist * error_dist + Kp_angle * error_angle
    # Only apply distance correction when aligned with wall
    if abs(error_angle) < 0.1:  # Only when reasonably aligned
        error_dist = clearance_wall - distance_to_wall
        forward = 0.03
    else:
        error_dist = 0
    # Slow down if obstacle in front
    front_dist = distances[180]

    if front_dist < 40 and counter >30:
        forward = 0.0
        print("Front object detected while wall following")
        rotation = 0.5 if side == "right" else -0.5  # Turn away
        wall_mode = False
        return {"forward": forward, "rotation": rotation}, side, wall_mode

    # Check if there is a space so we can stop moving
    wall_mode = True
    # i let 15 degree as gap to robot be able to pass the gap before stopping
    if last_side == "right" and distances[75] > dst_lim and distances[90] > dst_lim  and distances[105] > dst_lim:
        print("Right not wall detected")
        wall_mode = False
    elif last_side == "left" and distances[285] > dst_lim and distances[270] > dst_lim and distances[255] > dst_lim :
        print("Left not wall detected")
        wall_mode = False
    # limiting command values
    forward  = np.clip(forward, -1, 1)
    rotation = np.clip(rotation, -1, 1)
    print(error_angle, rotation)
    return {"forward": forward, "rotation": rotation}, side, wall_mode


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
    Kv = 0.5  # Attractive gain
    Kw = 0.2 # Angular gain
    Kobs = 4000 # Repulsive gain
    SAFE_DIST = 80.0  # Obstacle influence distance (meters)
    phi_max = 0.07  # Maximum angle for full speed
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
    
    # Calculate forward speed based on alignment
    module_gradient_F = np.linalg.norm(F_total)
    if abs(phi_R) < phi_max:
        vitesse = Kv * module_gradient_F
    else:
        vitesse = Kv * module_gradient_F * (phi_max / abs(phi_R))
    
    vitesse = np.clip(vitesse,-max_rot_speed,max_rot_speed)
    w_speed = np.clip(Kw * phi_R, -max_rot_speed, max_rot_speed)
    # print("Rotation: ",w_speed, " Phi_R: ",phi_R)
    return {"forward": vitesse, "rotation": w_speed}, goal_reachead

def local_control(current_pose, goal_pose):
    """
    Local control function to compute forward and rotation speeds
    based on the current pose and the goal pose.
    
    Parameters:
    - current_pose: [x, y, theta] nparray, current pose in world coordinates
    - goal_pose: [x, y, theta] nparray, goal pose in world coordinates
    
    Returns:
    - command: dict with "forward" and "rotation" keys
    """
    
    command = {"forward":forward, "rotation": rotation}
    return command


# class WallFollowerPID:
#     def __init__(self):
#         self.prev_error = 0.0
#         self.integral = 0.0
#         self.last_side = "Unknown"

#     def compute_command(self, lidar, dt=0.1, wall_clearance = 15, dst_lim = 60):
#         distances = np.array(lidar.get_sensor_values())
        
#         # Parameters
#         forward = 0.02
#         Kp_dist = 0.01     # Gain to correct distance error
#         Kp_angle = 0.5     # Gain to align with wall
#         min_idx = np.argmin(distances)
#         side = "left"
#         """
#                 180
#                 y+
#                 ↑
#         left      |    right
#                 |
#         y– ←----- 0 -----→ x+ 90
#             (robot)
#         """
#         if min_idx < 180:
#             side = "right"

#         if side == "right":
#             idx_a = 90  # Right
#             idx_b = 150  # Front-right
#         else:
#             idx_a = 270  # Left
#             idx_b = 240  # Front-left
        
#         #
#         # Get distances at two angles
#         dist_a = distances[idx_a]
#         dist_b = distances[idx_b]

#         # Compute angle of the wall (relative to robot)
#         alpha = np.arctan2(dist_b * np.cos(np.radians(idx_b - idx_a)) - dist_a,
#                         dist_b * np.sin(np.radians(idx_b - idx_a)))
        
#         # Estimate distance from wall (perpendicular)
#         distance_to_wall = dist_a * np.cos(alpha)

#         # Compute errors
#         error_dist = wall_clearance - distance_to_wall
#         error_angle = alpha

#         # Control rotation
#         #omega = Kp_dist * error_dist + Kp_angle * error_angle
#         error = error_dist + 50 * error_angle  # angle scaled to cm scale

#         # PID terms
#         Kp = 0.005
#         Ki = 0.0001
#         Kd = 0.002

#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt
#         self.prev_error = error

#         rotation = Kp * error + Ki * self.integral + Kd * derivative


#         # Slow down if obstacle in front
#         front_dist = distances[180]

#         if front_dist < 30:
#             forward = 0.0
#             rotation = 0.5 if side == "right" else -0.5  # Turn away

#         # Check if there is a space so we can stop moving
#         # check if radar goes beyond dst lim
#         wall_mode = True
#         if self.last_side == "right" and distances[90] > dst_lim:
#             wall_mode = False
#         elif self.last_side == "left" and distances[270] > dst_lim:
#             wall_mode = False
#         self.last_side = side
#         return {"forward": forward, "rotation": rotation}, wall_mode
