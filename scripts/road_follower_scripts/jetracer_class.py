import time
from typing import Union
import torch
from torch2trt import TRTModule

from utils import preprocess
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

from controllers.PID import PID
from controllers.MPC import MPC

class JetracerInitializer:
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
        self.camera_reading: float = 0.0

        self._last_print = time.time()