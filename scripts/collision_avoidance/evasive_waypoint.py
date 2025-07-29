import numpy as np
import matplotlib.pyplot as plt

def compute_global_evasion_waypoint(x0, y0, theta, d, lambda_):
    """
    Compute a global evasive waypoint.
    
    Parameters:
    - x0, y0: JetRacer's global position
    - theta: heading angle in radians
    - d: distance to obstacle straight ahead [m]
    - lambda_: evasion step perpendicular to heading [m] 
               (positive = step left, negative = step right)
    
    Returns:
    - obstacle_pos: (x, y) of the obstacle in global frame
    - evade_pos: (x, y) of the evasive waypoint in global frame
    """
    # Obstacle location in global frame
    x_obs = x0 + d * np.cos(theta)
    y_obs = y0 + d * np.sin(theta)

    # Step perpendicular to heading
    x_evade = x_obs - lambda_ * np.sin(theta)
    y_evade = y_obs + lambda_ * np.cos(theta)

    return (x_obs, y_obs), (x_evade, y_evade)