import time
from typing import Union
import torch
from torch2trt import TRTModule

from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

from controllers.PID import PID
from controllers.MPC import MPC

class JetracerInitializer:
    def __init__(
        self,
        model_path: str,
        controller: Union[PID, MPC],
        throttle_gain: float = 0.3,
        turning_away_duration: float = 0.5,
        turning_back_duration: float = 0.5,
        steering_away_value: float = 0.5,
        steering_back_value: float = -0.5,
        cam_w: int = 224,
        cam_h: int = 224,
        cam_fps: int = 65
    ):
        # Load vision model
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.state = True

        # Controller strategy (PID or MPC)
        self.ctrl = controller

        # Car setup
        self.car = NvidiaRacecar()
        self.car.steering_gain = -1.0    # no additional gain here
        self.car.throttle     = 0
        self.throttle_gain = throttle_gain

        # Obstracle avoidance parameters
        self.turning_away_duration = turning_away_duration
        self.turning_back_duration = turning_back_duration
        self.steering_away_value = steering_away_value
        self.steering_back_value = steering_back_value

        # Camera setup
        self.camera = CSICamera(width=cam_w, height=cam_h, capture_fps=cam_fps)
        self.camera_reading: float = 0.0
        self.preprocessed_camera: float = 0.0

        self._last_print = time.time()