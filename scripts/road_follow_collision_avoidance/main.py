#!/usr/bin/env python3

import numpy as np
from scripts.helpers.utils import preprocess
import torch
from typing import Tuple

from scripts.helpers.controller_setup import ControllerSetup
from scripts.helpers.jetracer_class import JetracerInitializer
from scripts.road_follow_collision_avoidance.road_follower_collision_class import RoadFollower
from scripts.collision_avoidance.collision_avoidance_class import CollisionAvoidance

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

# === Collision Avoidance parameters === ------------------------
turn_away_threshold = 0.5  
turning_away_duration = 0.5  # [s] time to turn away
turning_back_duration = 0.5  # [s] time to turn back
steering_away_value = 0.5    # steering value to turn away  
steering_back_value = -0.5   # steering value to turn back

# === Constants ===
road_follow_model_path      = 'scripts/trained_models/updated_model_trt.pth'
collision_avoidance_path = 'scripts/trained_models/best_model_resnet18_3class_trt.pth'
cam_w, cam_h    = 224, 224
cam_fps         = 65

class RoadFollowingCollisionAvoidance:
    def __init__(self) -> None:
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
            road_follow_model_path=road_follow_model_path,
            controller=ctrl,
            throttle_gain=throttle_gain,
            collision_avoidance_model_path=collision_avoidance_path,
            turning_away_duration=turning_away_duration,
            turning_back_duration=turning_back_duration,
            steering_away_value=steering_away_value,
            steering_back_value=steering_back_value,
            cam_w=cam_w,
            cam_h=cam_h,
            cam_fps=cam_fps
        )

        self.jetracer = jet
        self.road_follower = RoadFollower(jetracer=jet)
        self.collision_avoidance = CollisionAvoidance(jetracer=jet)

    def get_correct_camera_vector(self, controller: str = "pid") -> float:
        if controller == "mpc":
            return -1 * camera_scale
        else:
            return 1

    def get_collision_probabilities(self) -> Tuple[float, float, float]:
        self.jetracer.camera_reading = self.jetracer.camera.read()
        self.jetracer.preprocessed_camera = preprocess(self.jetracer.camera_reading).half()
        action_output = self.jetracer.model_collision(self.jetracer.preprocessed_camera)
        probabilities = torch.softmax(action_output, dim=1) #.cpu().numpy().flatten()
        p_line_follow = probabilities[0, 0].item()
        p_left    = probabilities[0, 1].item()
        p_right   = probabilities[0, 2].item()

        return p_line_follow, p_left, p_right


    def run(self) -> None:
        camera_vector = self.get_correct_camera_vector(controller=controller)
        
        while True:
            p_line_follow, p_left, p_right = self.get_collision_probabilities()
            
            if p_line_follow <= turn_away_threshold: # Turn away if the model predicts a collision
                ''' If the model predicts a collision, turn away from the road
                    and then turn back to the road.
                '''
                if p_left > p_right:
                    print("Turning left to avoid collision...")
                    self.collision_avoidance.left_turn()
                else:
                    print("Turning right to avoid collision...")
                    self.collision_avoidance.right_turn()
            else: # Follow the road
                self.road_follower.run(camera_scale=camera_vector)


if __name__ == '__main__':
    system = RoadFollowingCollisionAvoidance()
    system.run()
