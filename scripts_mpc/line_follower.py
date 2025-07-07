import time
import numpy as np
import torch
from torch2trt import TRTModule

from utils import preprocess
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

# Import controllers from separate modules
from controllers.PID import PID
from controllers.MPC import MPC

# === Constants ===
MODEL_PATH      = 'trained_models/updated_model_trt.pth'
CAM_W, CAM_H    = 224, 224
CAM_FPS         = 65
THROTTLE_GAIN   = 0.7
PRINT_INTERVAL  = 2.0

# === PID Parameters ===
PID_KP          = 0.85
PID_KI          = 0.1
PID_KD          = 0.2
PID_INTEGRAL_RESET = 0.00
PID_DELAY       = 0.0

# === MPC Parameters ===
N               = 30          # horizon
DT              = 0.05        # timestep
MAX_STEER       = np.pi/4     # 45
W_Y             = 10          # CTE weight
W_DELTA         = 1           # steering weight
STEER_GAIN_MPC  = 1

class RoadFollower:
    def __init__(
        self,
        model_path: str,
        controller,
        throttle_gain: float,
        cam_w: int,
        cam_h: int,
        cam_fps: int,
        print_interval: float = PRINT_INTERVAL,
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
        self.car.throttle     = throttle_gain

        # Camera setup
        self.camera = CSICamera(width=cam_w, height=cam_h, capture_fps=cam_fps)

        self._last_print = time.time()
        self.print_interval = print_interval

    def run(self):
        try:
            while True:
                # 1) Capture & preprocess
                frame = self.camera.read()
                tensor = preprocess(frame).half()

                # 2) Inference
                with torch.no_grad():
                    out = self.model(tensor).cpu().numpy().flatten()
                x_norm = float(out[0])

                # 3) Compute steering via selected controller
                steer = self.ctrl.compute_steering(x_norm)
                self.car.steering = steer
                # Set car speed
                '''k = 5
                self.car.throttle = 0.65 + 1 * np.exp(-abs(steer) * k)'''
                # self.car.throttle = -0.25 * np.abs(steer) + 1
                # shape = 20
                # min_speed = 0.6
                # max_speed = 0.85
                # self.car.throttle = (max_speed - min_speed) * np.exp(-shape * (steer**2)) + min_speed

                # 4) Periodic logging
                now = time.time()
                if now - self._last_print >= self.print_interval:
                    print(f"Apex x: {x_norm:.3f}, Steering: {steer:.3f}, Speed: {self.car.throttle:.3f}")
                    self._last_print = now
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping motors...")
            # Stop the car safely
            self.car.steering = 0
            self.car.throttle = 0
            print("Motors stopped.")


def main():
    # Choose one of the controllers:
    pid_ctrl = PID(
        Kp=PID_KP,
        Ki=PID_KI,
        Kd=PID_KD,
        integral_reset=PID_INTEGRAL_RESET,
        delay=PID_DELAY
    )
    mpc_ctrl = MPC(
        N=N,
        dt=DT,
        max_steer=MAX_STEER,
        w_y=W_Y,
        w_delta=W_DELTA,
        steer_gain=STEER_GAIN_MPC
    )

    # Swap between PID and MPC here:
    controller = pid_ctrl
    #controller = mpc_ctrl

    follower = RoadFollower(
        model_path=MODEL_PATH,
        controller=controller,
        throttle_gain=THROTTLE_GAIN,
        cam_w=CAM_W, cam_h=CAM_H, cam_fps=CAM_FPS
    )
    follower.run()

if __name__ == '__main__':
    main()
