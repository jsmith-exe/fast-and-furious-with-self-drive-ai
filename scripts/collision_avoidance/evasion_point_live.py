import rospy
from nav_msgs.msg import Odometry
import numpy as np
from scripts.collision_avoidance.evasive_waypoint import compute_global_evasion_waypoint
from scripts.collision_avoidance.object_detection_follow import ObjectLiveDetector

from flask import Flask, Response, jsonify
import threading
import cv2


class EvasionPointStreamer:
    def __init__(self, odom_topic='/odom', rate_hz=2):
        self.detector = ObjectLiveDetector()
        self.detector.start()

        self.app = Flask(__name__)
        self._setup_routes()
        self.flask_thread = threading.Thread(target=self._run_flask, daemon=True)
        self.flask_thread.start()

        self.odom_topic = odom_topic
        self.rate = rospy.Rate(rate_hz)
        self.odom_msg = None
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        print(f"Subscribed to {self.odom_topic} for car position.")

    def _setup_routes(self):
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/evasion_data')
        def evasion_data():
            dist, bbox, _ = self.detector.get_latest()
            return jsonify({
                'distance': dist,
                'bbox': bbox
            })

    def _run_flask(self):
        self.app.run(host='0.0.0.0', port=5000)

    def _gen_frames(self):
        while True:
            dist, bbox, frame = self.detector.get_latest()
            if frame is None:
                continue
            annotated = frame.copy()
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"Dist: {dist:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def odom_callback(self, msg):
        self.odom_msg = msg

    def get_car_position(self):
        if self.odom_msg is None:
            return None, None, None
        pos = self.odom_msg.pose.pose.position
        x = pos.x
        y = pos.y
        q = self.odom_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        return x, y, theta

    def get_object_position(self):
        dist, bbox, _ = self.detector.get_latest()
        return dist, bbox

    def process_evasion_point(self, x, y, theta, distance, lambda_=0.22):
        if distance is not None:
            obstacle_pos, evade_pos = compute_global_evasion_waypoint(x, y, theta, distance, lambda_)

             # Step 2: Compute return point (same x as start, y = y_evade + distance)
            return_x = obstacle_pos[0] + distance
            return_y = obstacle_pos[1]
            return_pos = (return_x, return_y)
            return obstacle_pos, evade_pos, return_pos
        return None, None, None


    def compute_turn_angle_right_positive(self, x: float, y: float, theta: float, target_x: float, target_y: float) -> float:
    
        theta_rad = theta # Assuming theta is in radians
        dx = target_x - x
        dy = target_y - y
        target_angle = np.arctan2(dy, dx)
        angle_diff_rad = target_angle - theta_rad
        angle_diff_rad = (angle_diff_rad + np.pi) % (2 * np.pi) - np.pi
        signed_angle_deg = -np.rad2deg(angle_diff_rad)  # Flip sign: left = neg, right = pos
        return signed_angle_deg


    def stream(self):
        while not rospy.is_shutdown():
            x, y, theta = self.get_car_position()
            distance, bbox = self.get_object_position()
            if x is not None:
                print(f"Car position: (x={x:.2f}, y={y:.2f}, theta={np.rad2deg(theta):.1f} deg)")
            else:
                print("No odom message received yet.")
            if bbox:
                print(f"Object bbox: {bbox}, Distance: {distance:.2f} m")
            else:
                print("No object detected by camera.")
            if x is not None and y is not None and theta is not None and distance is not None:
                obstacle_pos, evade_pos, return_pos = self.process_evasion_point(x, y, theta, distance+0.1)
                if obstacle_pos and evade_pos:
                    print(f"Obstacle: (x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f}), Distance: {distance:.2f} m")
                    print(f"Evasion point: (x={evade_pos[0]:.2f}, y={evade_pos[1]:.2f})")
                    print(f"Return point:  (x={return_pos[0]:.2f}, y={return_pos[1]:.2f})")

                    angle_to_evade = self.compute_turn_angle_right_positive(x, y, theta, evade_pos[0], evade_pos[1])
                    angle_to_return = self.compute_turn_angle_right_positive(evade_pos[0], evade_pos[1], theta, return_pos[0], return_pos[1])

                    print(f"Evasion turn angle: {angle_to_evade:+.2f} degrees")
                    print(f"Return turn angle:  {angle_to_return:+.2f} degrees\n")
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('evasion_point_streamer')
    streamer = EvasionPointStreamer()
    streamer.stream()
