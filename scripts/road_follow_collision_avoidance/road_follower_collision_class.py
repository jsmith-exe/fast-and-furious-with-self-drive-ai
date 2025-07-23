import numpy as np
import torch

from scripts.helpers.utils import preprocess
from scripts.helpers.jetracer_class import JetracerInitializer

class RoadFollower:
    def __init__(self, jetracer: JetracerInitializer) -> None:
        self.jetracer = jetracer
        self.car = jetracer.car

        # Camera and model setup
        self.camera = jetracer.camera
        self.model = jetracer.model

        # Controller strategy
        self.ctrl = jetracer.ctrl

    def get_camera_offset(self, camera_scale) -> float:
        return self.jetracer.camera_reading / camera_scale

    def process_steering_value(self, steer_radians: float) -> float:
        steer_degrees = np.rad2deg(steer_radians)
        return steer_degrees / np.rad2deg(self.ctrl.u_bounds[1])

    def run(self, camera_scale: float = 1.0) -> None:
        try:
            # 1) Capture & preprocess
            frame = self.camera.read()
            tensor = preprocess(frame).half()

            # 2) Inference
            with torch.no_grad():
                out = self.model(tensor).cpu().numpy().flatten()

            self.jetracer.camera_reading = float(out[0])

            cte = self.get_camera_offset(camera_scale=camera_scale)

            # 3) Compute steering via selected controller
            steer, latency = self.ctrl.compute_steering(error=cte)
            # 3.1) Process steering value
            if camera_scale != 1:
                steer = self.process_steering_value(steer_radians=steer)

            steer_clipped = np.clip(steer, -1, 1)
            self.car.steering = steer_clipped

            self.car.throttle = self.jetracer.throttle_gain

            # 4) Periodic Logging
            print(f"camera: {self.jetracer.camera_reading:+.3f}, steering: {steer_clipped:+.3f}, latency: {latency:.3f} ms, throttle: {self.car.throttle:.3f}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping motors...")
            # Stop the car safely
            self.car.steering = 0
            self.car.throttle = 0
            print("Motors stopped.")