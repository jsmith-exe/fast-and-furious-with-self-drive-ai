#!/usr/bin/env python3
"""
Entry‑point that launches:
  • Road‑following (daemon thread)
  • Object detection + Flask MJPEG/JSON API (thread inside detector)
  • ROS + evasion‑point calculations (main thread)
All while sharing ONE CSI camera created in JetracerInitializer.
"""
import threading, rospy, numpy as np
from scripts.helpers.controller_setup import ControllerSetup
from scripts.helpers.jetracer_class import JetracerInitializer
from scripts.road_follower.road_follower_class_object import RoadFollower
from scripts.collision_avoidance.object_detection_follow import ObjectLiveDetector
from scripts.collision_avoidance.evasion_point_follow import EvasionPointStreamer

# ───────── configuration ─────────
controller       = "pid"
throttle_gain    = 0.0
camera_scale     = 4.44
KP, KI, KD       = 0.85, 0.10, 0.20
integral_reset   = 0.01
N, dt            = 30, 0.1
max_steer        = 30         # deg
w_y, w_delta     = 10.0, 0.5
model_path       = "scripts/trained_models/updated_model_trt.pth"
cam_w, cam_h     = 224, 224
cam_fps          = 65         # safer than 65
# ─────────────────────────────────

def get_correct_camera_vector(controller: str = "pid") -> float:
    if controller == "mpc":
        return -1 * camera_scale
    else:
        return 1

def main():
    camera_lock = threading.Lock()

    # 1. controller + JetRacer (creates ONE camera)
    ctrl = ControllerSetup(
        kp=KP, ki=KI, kd=KD, integral_reset=integral_reset,
        max_steer=np.deg2rad(max_steer),
        N=N, dt=dt, w_y=w_y, w_delta=w_delta,
    ).get_controller(controller=controller)

    jet = JetracerInitializer(
        road_follow_model_path=model_path,
        controller=ctrl,
        throttle_gain=throttle_gain,
        cam_w=cam_w, cam_h=cam_h, cam_fps=cam_fps,
    )

    # 2. road‑follower — separate daemon thread
    follower = RoadFollower(jetracer=jet, camera_lock=camera_lock)

    rf_thread = threading.Thread(
        target=follower.run,                  # pass the function itself
        kwargs={"camera_scale": get_correct_camera_vector(controller)},
        daemon=True,
    )
    rf_thread.start()   

    # 3. object detector — shares the *same* camera
    detector = ObjectLiveDetector(camera=jet.camera,
                                  width=cam_w, height=cam_h, capture_fps=cam_fps,
                                  camera_lock=camera_lock)
    detector.start()

    # 4. ROS + evasion streamer — keep main thread alive
    rospy.init_node("evasion_point_streamer", anonymous=True)
    EvasionPointStreamer(detector=detector, jetracer=jet).stream()

if __name__ == "__main__":
    main()
