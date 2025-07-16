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

throttle_gain   = 0.8
camera_scale = 4

# === PID parameters === ---------------------------------------
KP = 2.0
KI = 0.1
KD = 0.2
integral_reset = None

# === MPC parameters === ---------------------------------------
N  = 40          # prediction horizon
dt = 0.05        # [s] integration step
v  = 3         # [m/s] constant forward speed
max_steer = np.deg2rad(40)  # [rad] hard steering limit
max_cte = None # max cte

# Cost weights
w_y     = 5.0   # weight on lateral error (y)
w_delta = 0.5    # weight on steering usage (delta)

# === Constants ===
model_path      = 'trained_models/updated_model_trt.pth'
cam_w, cam_h    = 224, 224
cam_fps         = 65
print_interval  = 2.0

def main() -> None:
    controller_setup = ControllerSetup(
        kp=KP,
        ki=KI,
        kd=KD,
        integral_reset=integral_reset,
        max_steer=max_steer,
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
    follow.run(camera_scale=camera_scale)


if __name__ == '__main__':
    main()
