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


d = 0.8                        # Distance to obstacle [m]
lambda_ = 0.5                  # Sideways evasion distance [m]








if __name__ == "__main__":
    # === Parameters ===
    x0, y0 = 4.0, 2.75              # JetRacer's global position
    theta = np.deg2rad(-155)         # JetRacer heading in radians
    d = 1.2                        # Distance to obstacle [m]
    lambda_ = 0.5                  # Sideways evasion distance [m]

    # === Compute positions ===
    obstacle_pos, evade_pos = compute_global_evasion_waypoint(x0, y0, theta, d, lambda_)

    # === Plotting ===
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(2.5, 4.5)
    ax.set_ylim(1.5, 3.5)

    # Plot JetRacer at (x0, y0)
    ax.plot(x0, y0, 'ro', label='JetRacer')
    ax.arrow(x0, y0, np.cos(theta)*0.3, np.sin(theta)*0.3, head_width=0.05, color='r', label='Heading')

    # Plot obstacle
    ax.plot(*obstacle_pos, 'ks', label='Obstacle')
    ax.text(*obstacle_pos, ' Obstacle', va='bottom', ha='left')

    # Plot evasive waypoint
    ax.plot(*evade_pos, 'go', label='Evasive Waypoint')
    ax.text(*evade_pos, ' Evade', va='bottom', ha='left')

    # Connect points with dashed lines
    ax.plot([x0, obstacle_pos[0]], [y0, obstacle_pos[1]], 'k--', alpha=0.6)
    ax.plot([obstacle_pos[0], evade_pos[0]], [obstacle_pos[1], evade_pos[1]], 'g--', alpha=0.6)

    # Final touches
    ax.legend()
    plt.title("Global Evasive Waypoint Calculation")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.show()
