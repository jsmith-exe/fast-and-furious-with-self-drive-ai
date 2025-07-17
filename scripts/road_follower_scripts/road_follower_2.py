#!/usr/bin/env python3

import numpy as np
import torch
from torch2trt import TRTModule

from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

from scripts.helpers.controller_setup import ControllerSetup
from road_follower_class import RoadFollower

# === Select Sim Modes ---------------------------------------------------------
''' controller types: 
    - pid : use a Proportional Integral Differential Controller
    - mpc : use Model Predictive Controller              
'''
controller = "mpc"

throttle_gain   = 0.3
camera_scale = 4.44

# === PID parameters === ---------------------------------------
KP = 1.0
KI = 0.1
KD = 0.2
integral_reset = 0.01

# === MPC parameters === ---------------------------------------
N  = 20          # prediction horizon
dt = 0.1        # [s] integration step
max_steer = 35  # Degree hard steering limit
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
    follow = RoadFollower(
        model_path=model_path,
        controller=ctrl,
        throttle_gain=throttle_gain,
        cam_w=cam_w,
        cam_h=cam_h,
        cam_fps=cam_fps
    )

    camera_vector = get_correct_camera_vector(controller=controller)
    follow.run(camera_scale=camera_vector)


if __name__ == '__main__':
    main()
