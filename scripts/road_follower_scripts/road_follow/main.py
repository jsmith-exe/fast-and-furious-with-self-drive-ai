#!/usr/bin/env python3

import numpy as np

from scripts.helpers.controller_setup import ControllerSetup
from scripts.road_follower_scripts.jetracer_class import JetracerInitializer
from road_follower_class import RoadFollower

# === Select Sim Modes ---------------------------------------------------------
''' controller types: 
    - pid : use a Proportional Integral Differential Controller
    - mpc : use Model Predictive Controller              
'''
controller = "pid"

throttle_gain   = 0.3
camera_scale = 4.44

# === PID parameters === ---------------------------------------
KP = 0.85 # 1
KI = 0.1
KD = 0.2
integral_reset = 0.01

# === MPC parameters === ---------------------------------------
N  = 20          # prediction horizon
dt = 0.1        # [s] integration step
max_steer = 30  # Degree hard steering limit
max_cte = None # max cte

# Cost weights
w_y     = 10.0   # weight on lateral error (y)
w_delta = 0.5    # weight on steering usage (delta)

# === Constants ===
model_path      = 'scripts/road_follower_scripts/trained_models/updated_model_trt.pth'
cam_w, cam_h    = 224, 224
cam_fps         = 65

def get_correct_camera_vector(controller: str = "pid") -> float:
    if controller == "mpc":
        return -1 * camera_scale
    else:
        return 1

def main() -> None:
    controller_setup = ControllerSetup(
        kp=KP,
        ki=KI,
        kd=KD,
        integral_reset=integral_reset,
        max_steer=np.deg2rad(max_steer),
        N=N,
        dt=dt,
        w_y=w_y,
        w_delta=w_delta,
    )
    ctrl = controller_setup.get_controller(controller=controller)
    jet = JetracerInitializer(
        road_follow_model_path=model_path,
        controller=ctrl,
        throttle_gain=throttle_gain,
        cam_w=cam_w,
        cam_h=cam_h,
        cam_fps=cam_fps
    )
    road_follower = RoadFollower(jetracer=jet)

    camera_vector = get_correct_camera_vector(controller=controller)
    road_follower.run(camera_scale=camera_vector)


if __name__ == '__main__':
    main()
