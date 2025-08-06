#!/usr/bin/env python
"""
lane_detector.py
ROS node to detect black lane edges and compute lateral deviation & yaw relative to camera centerline, streaming MJPEG
Computes lateral & yaw for lane midline and, when an obstacle is detected, for evasion line.
"""

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import collections
import numpy as np
import cv2
from threading import Thread, Lock
from flask import Flask, Response

# Flask app for MJPEG streaming
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()

def gen_frames():
    global latest_frame
    while not rospy.is_shutdown():
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

class LineHeadingNode:
    def __init__(self):
        rospy.init_node('line_heading_node')
        self.bridge = CvBridge()

        # Parameters
        self.camera_height   = rospy.get_param('~camera_height', 0.3)
        self.cam_fps         = rospy.get_param('~cam_fps', 30)
        self.black_thresh    = rospy.get_param('~black_thresh', 90)
        self.display_width_m = rospy.get_param('~display_width_m', 0.16)
        # Object-detection
        self.lower_hsv          = np.array(rospy.get_param('~obj_lower_hsv', [10, 40, 50]))
        self.upper_hsv          = np.array(rospy.get_param('~obj_upper_hsv', [40, 140, 150]))
        self.ref_dist           = rospy.get_param('~obj_ref_dist', 0.20)
        self.ref_width          = rospy.get_param('~obj_ref_width', 45)
        self.alpha              = rospy.get_param('~obj_alpha', 0.2)
        self.min_area           = rospy.get_param('~obj_min_area', 500)
        self.bbox_history       = collections.deque(maxlen=5)
        self.filtered_distance  = None

        # Camera tilt
        tilt_deg = rospy.get_param('~camera_tilt_from_vertical_deg', 45)
        tilt_rad = np.deg2rad(tilt_deg)
        self.camera_pitch = tilt_rad - (np.pi/2)

        # State
        self.K          = None
        self.dist       = None
        self.img_w      = None
        self.img_h      = None
        self.roi        = None
        self.last_stamp = rospy.Time(0)
        self.min_dt     = rospy.Duration(1.0 / self.cam_fps)

        # ROS interfaces
        self.pub = rospy.Publisher('/line/offset_yaw', Float32MultiArray, queue_size=1)
        rospy.Subscriber('/csi_cam_0/camera_info', CameraInfo, self.cam_info_cb)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.image_cb, queue_size=1)

        # Start MJPEG server
        flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000,
                                                     threaded=True, use_reloader=False))
        flask_thread.daemon = True
        flask_thread.start()

        rospy.loginfo("[LineHeadingNode] MJPEG streaming on http://localhost:5000/video_feed")
        rospy.spin()

    def cam_info_cb(self, msg):
        if self.K is None:
            self.K    = np.array(msg.K).reshape(3,3)
            self.dist = np.array(msg.D)
            self.img_w = msg.width
            self.img_h = msg.height
            self.roi   = msg.roi

    def image_cb(self, msg):
        # Throttle and wait for camera calibration
        stamp = msg.header.stamp or rospy.Time.now()
        if stamp - self.last_stamp < self.min_dt or self.K is None:
            return
        self.last_stamp = stamp

        # Convert & undistort
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        frame = cv2.resize(cv2.undistort(img, self.K, self.dist), (self.img_w, self.img_h))

        # Crop ROI
        if self.roi and self.roi.height>0 and self.roi.width>0:
            x0, y0 = self.roi.x_offset, self.roi.y_offset
            frame = frame[y0:y0+self.roi.height, x0:x0+self.roi.width]

        # Threshold black lane
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_black = cv2.threshold(gray, self.black_thresh, 255, cv2.THRESH_BINARY_INV)

        # Sample midpoints
        h, w = mask_black.shape
        mid_pts = []
        for y in range(int(h*0.3), int(h*0.55), 10):
            cols = np.where(mask_black[y] > 0)[0]
            if cols.size:
                mid_pts.append((int(np.mean(cols)), y))

        # # Obstacle detection first
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv_blur = cv2.GaussianBlur(hsv, (7,7), 0)
        # mask_obj = cv2.inRange(hsv_blur, self.lower_hsv, self.upper_hsv)
        # kernel = np.ones((5,5), np.uint8)
        # mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel)
        # mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)
        # contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # obstacle = False
        # bbox = None
        # evasion_pts = None
        # if contours:
        #     largest = max(contours, key=cv2.contourArea)
        #     if cv2.contourArea(largest) > self.min_area:
        #         obstacle = True
        #         x, y, w_box, h_box = cv2.boundingRect(largest)
        #         self.bbox_history.append((x,y,w_box,h_box))
        #         if len(self.bbox_history) > 1:
        #             x     = int(np.mean([b[0] for b in self.bbox_history]))
        #             y     = int(np.mean([b[1] for b in self.bbox_history]))
        #             w_box = int(np.mean([b[2] for b in self.bbox_history]))
        #             h_box = int(np.mean([b[3] for b in self.bbox_history]))
        #         # Define evasion line points
        #         h_img = frame.shape[0]
        #         offset = rospy.get_param('~evasion_offset_px', 200)
        #         sample_ys = list(range(int(h_img*0.3), int(h_img*0.55), 10))
        #         evasion_pts = [(x - offset, yy) for yy in sample_ys]

        # # Compute metrics
        # if obstacle and evasion_pts:
        #     # Use far and near evasion points relative to camera centerline
        #     center_x = w / 2.0
        #     known_distance = 0.1; known_px = 192
        #     multi = known_distance / known_px
        #     fx = self.K[0,0]
        #     camera_tilt = np.deg2rad(6)
        #     # near point at maximum y (bottom of sampled evasion_pts)
        #     px_near, py_near = evasion_pts[-1]
        #     lateral_px = px_near - center_x
        #     lateral_m = lateral_px * multi
        #     # far point at minimum y (top of sampled evasion_pts)
        #     px_far, py_far = evasion_pts[0]
        #     dx = px_far - px_near
        #     pixel_angle = np.arctan2(dx, fx)
        #     scale = (lateral_m / 1.8) / -0.04
        #     yaw_rad = pixel_angle - (camera_tilt * scale)
        #     # publish evasion metrics
        #     self.pub.publish(Float32MultiArray(data=[lateral_m, yaw_rad]))
        #     # annotate
        #     cv2.putText(frame, f"e_lat{lateral_m:.3f}m, e_yaw={np.rad2deg(yaw_rad):.2f}deg",
        #                 (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if len(mid_pts) >= 2:
            # No obstacle → lane midline metrics
            mid_sorted = sorted(mid_pts, key=lambda p: p[1])
            u_far, v_far = mid_sorted[0]
            u_near, v_near = mid_sorted[-1]
            center_x = w / 2.0
            known_distance = 0.1; known_px = 192
            multi = known_distance / known_px
            lateral_px = u_near - center_x
            lateral_m = lateral_px * multi
            fx = self.K[0,0]
            dx = u_far - u_near
            pixel_angle = np.arctan2(dx, fx)
            camera_tilt = np.deg2rad(6)
            scale = (lateral_m / 1.8) / -0.04
            yaw_rad = pixel_angle - (camera_tilt * scale)
            # publish lane metrics
            self.pub.publish(Float32MultiArray(data=[lateral_m, yaw_rad]))
            # annotate
            cv2.putText(frame, f"lat{lateral_m:.3f}m, yaw={np.rad2deg(yaw_rad):.2f}deg",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Draw points
        if False:
            # obstacle and evasion_pts:
            for px, py in evasion_pts:
                cv2.circle(frame, (int(px), int(py)), 5, (0,255,0), -1)
        else:
            for u, v in mid_pts:
                cv2.circle(frame, (u, v), 3, (0,255,0), -1)

        # Stream
        global latest_frame
        with frame_lock:
            latest_frame = frame.copy()

if __name__=='__main__':
    try:
        LineHeadingNode()
    except rospy.ROSInterruptException:
        pass
