import numpy as np
from evasive_waypoints import compute_global_evasion_waypoint

# Simulate a sequence of car positions and obstacle distances
car_states = [
    # (x0, y0, theta_deg, d, lambda_)
    (3.0, 2.0, 0,    1.0, 0.5),
    (3.2, 2.1, 15,   0.8, 0.5),
    (3.4, 2.2, 30,   0.6, 0.5),
    (3.6, 2.3, 45,   0.5, 0.5),
    (3.8, 2.4, 60,   0.4, 0.5),
]

print("Simulating evasive waypoint calculation for a moving car:")
for i, (x0, y0, theta_deg, d, lambda_) in enumerate(car_states):
    theta = np.deg2rad(theta_deg)
    obstacle_pos, evade_pos = compute_global_evasion_waypoint(x0, y0, theta, d, lambda_)
    print(f"Step {i+1}:")
    print(f"  Car position:    (x={x0:.2f}, y={y0:.2f}), heading={theta_deg} deg")
    print(f"  Obstacle dist:   {d:.2f} m")
    print(f"  Evasion step:    {lambda_:.2f} m")
    print(f"  Obstacle pos:    (x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f})")
    print(f"  Evasive waypoint:(x={evade_pos[0]:.2f}, y={evade_pos[1]:.2f})\n")
