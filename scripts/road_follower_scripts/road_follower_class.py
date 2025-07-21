import time
import numpy as np
from typing import Union
import torch
from torch2trt import TRTModule

from utils import preprocess
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

from controllers.PID import PID
from controllers.MPC import MPC

class RoadFollower:
    def __init__(
        self,
        model_path: str,
        controller: Union[PID, MPC],
        throttle_gain: float,
        cam_w: int,
        cam_h: int,
        cam_fps: int
    ):
        # Load vision model
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Controller strategy (PID or MPC)
        self.ctrl = controller

        # Car setup
        self.car = NvidiaRacecar()
        self.car.steering_gain = -1.0    # no additional gain here
        self.car.throttle     = 0
        self.throttle_gain = throttle_gain

        # Camera setup
        self.camera = CSICamera(width=cam_w, height=cam_h, capture_fps=cam_fps)
        self.camera_reading = 0

        self._last_print = time.time()

    def get_camera_offset(self, camera_scale) -> float:
        return self.camera_reading / camera_scale

    def process_steering_value(self, steer_radians: float) -> float:
        steer_degrees = np.rad2deg(steer_radians)
        return steer_degrees / np.rad2deg(self.ctrl.u_bounds[1])

    def run(self, camera_scale: float) -> None:
        try:
            while True:
                # 1) Capture & preprocess
                frame = self.camera.read()
                tensor = preprocess(frame).half()

                # 2) Inference
                with torch.no_grad():
                    out = self.model(tensor).cpu().numpy().flatten()
                self.camera_reading = float(out[0])

                cte = self.get_camera_offset(camera_scale=camera_scale)

                # 3) Compute steering via selected controller
                steer, latency = self.ctrl.compute_steering(error=cte)
                steer_rad = steer
                if camera_scale is not 1:
                    steer = self.process_steering_value(steer_radians=steer)
                steer_clipped = np.clip(steer, -1, 1)
                self.car.steering = steer_clipped

                self.car.throttle = self.throttle_gain

                # 4) Periodic Logging
                print(f"camera: {self.camera_reading:+.3f}, steering: {steer_clipped:+.3f}, latency: {latency:.3f} ms, throttle: {self.car.throttle:.3f}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping motors...")
            # Stop the car safely
            self.car.steering = 0
            self.car.throttle = 0
            print("Motors stopped.")