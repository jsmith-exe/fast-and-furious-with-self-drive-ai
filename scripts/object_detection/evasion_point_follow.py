import threading, numpy as np, rospy, cv2
from nav_msgs.msg import Odometry
from flask import Flask, Response, jsonify

# ── project helpers ────────────────────────────────────────────────────────
from scripts.collision_avoidance.object_detection_follow import (
    ObjectLiveDetector,
    create_flask_app,                       # <- factory we already wrote
)
from scripts.collision_avoidance.evasive_waypoint import compute_global_evasion_waypoint
from scripts.road_follower.jetracer_class import JetracerInitializer
# ───────────────────────────────────────────────────────────────────────────


class EvasionPointStreamer:
    """
    Streams pose, steering, throttle, obstacle distance AND serves
    /video_feed + /evasion_data automatically.

    Parameters
    ----------
    detector : ObjectLiveDetector
        Must share the *same* CSI camera instance used elsewhere.
    jetracer : JetracerInitializer
        Gives access to current steering / throttle.
    start_stream : bool, default True
        If False, caller can start its own Flask server later.
    odom_topic : str
    rate_hz : int
    """

    def __init__(
        self,
        detector: ObjectLiveDetector,
        jetracer: JetracerInitializer,
        *,
        start_stream: bool = True,
        odom_topic: str = "/odom",
        rate_hz: int = 2,
    ):
        self.detector = detector
        self.car = jetracer.car

        # ───────── ROS pose subscriber ─────────
        self.x = self.y = self.theta = None
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb)
        self.rate = rospy.Rate(rate_hz)

        # ───────── optional Flask streamer ─────
        if start_stream:
            self.app = create_flask_app(detector)
            self._flask_thread = threading.Thread(
                target=lambda: self.app.run(
                    host="0.0.0.0", port=5000, threaded=True, use_reloader=False
                ),
                daemon=True,
            )
            self._flask_thread.start()

    # ─────────── ROS callback ───────────
    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = np.arctan2(siny_cosp, cosy_cosp)

    # ─────────── helper: compute evasion / return ───────────
    def _process_evasion(self, distance, lateral_ratio=0.1):
        obstacle_pos, evade_pos = compute_global_evasion_waypoint(
            self.x, self.y, self.theta, distance, lateral_ratio
        )
        return_pos = (self.x, evade_pos[1] + distance)
        return obstacle_pos, evade_pos, return_pos

    # ─────────── main loop ───────────
    def stream(self):
        while not rospy.is_shutdown():
            dist, bbox, _ = self.detector.get_latest()

            if self.x is not None:
                print(f"[POSE]  x={self.x:6.2f}  y={self.y:6.2f}  θ={np.rad2deg(self.theta):6.1f}°")
            else:
                print("[POSE]  – no /odom yet –")

            if bbox:
                print(f"[OBJ ]  bbox={bbox}  dist={dist:4.2f} m")
            else:
                print("[OBJ ]  – no obstacle –")

            print(f"[CTRL]  steering={self.car.steering:+5.2f}  throttle={self.car.throttle:+4.2f}")

            if self.x is not None and dist is not None and bbox:
                obstacle_pos, evade_pos, return_pos = self._process_evasion(dist)
                print(
                    f"[EVAS] obstacle=({obstacle_pos[0]:.2f},{obstacle_pos[1]:.2f}) → "
                    f"evade=({evade_pos[0]:.2f},{evade_pos[1]:.2f})  "
                    f"return=({return_pos[0]:.2f},{return_pos[1]:.2f})"
                )

            self.rate.sleep()


# ───────── stand‑alone test ─────────
if __name__ == "__main__":
    rospy.init_node("evasion_point_streamer", anonymous=True)

    # Build ONE JetRacer (creates the shared camera)
    jet = JetracerInitializer(
        road_follow_model_path="scripts/trained_models/updated_model_trt.pth",
        controller=None,
        throttle_gain=0,
    )

    detector = ObjectLiveDetector(camera=jet.camera)
    detector.start()

    try:
        EvasionPointStreamer(detector=detector, jetracer=jet).stream()
    finally:
        detector.stop()
