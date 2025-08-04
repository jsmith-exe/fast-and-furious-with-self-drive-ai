#!/usr/bin/env python
"""
line_heading_node.py
ROS node to compute and stream annotated line detection over HTTP
Method: Hough-based line detection with ground projection, streaming MJPEG on localhost:5000
"""

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from threading import Thread, Lock
from flask import Flask, Response

# Flask app for video streaming
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
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

class LineHeadingNode:
    def __init__(self):
        rospy.init_node('line_heading_node')
        self.bridge = CvBridge()

                # Node parameters
        self.camera_height = rospy.get_param('~camera_height', 0.3)    # meters
        self.camera_pitch = rospy.get_param('~camera_pitch', 0.0)     # radians
        self.cam_fps = rospy.get_param('~cam_fps', 30)                 # processing rate
        # Low-pass filter parameter for edge smoothing
        self.low_pass_alpha = rospy.get_param('~low_pass_alpha', 0.2)
        # Initialize previous filtered positions
        self.prev_left = None
        self.prev_right = None

        # Hough-line parameters
        self.canny_min    = rospy.get_param('~canny_min', 50)
        self.canny_max    = rospy.get_param('~canny_max', 150)
        self.hough_thresh = rospy.get_param('~hough_thresh', 50)
        self.min_line_len = rospy.get_param('~min_line_len', 50)
        self.max_line_gap = rospy.get_param('~max_line_gap', 20)
        # Visualization offsets in pixels for left/right rails
        self.line_offset_px = rospy.get_param('~line_offset_px', 20)

        # Camera intrinsics/distortion
        self.K = None
        self.dist = None
        self.img_w = None
        self.img_h = None
        self.roi = None
        self.binning_x = 1
        self.binning_y = 1

        # Throttle processing
        self.last_stamp = rospy.Time(0)
        self.min_dt = rospy.Duration(1.0 / float(self.cam_fps))

        # ROS interfaces
        rospy.Subscriber('/csi_cam_0/camera_info', CameraInfo, self.cam_info_cb)
        rospy.Subscriber('/csi_cam_0/image_raw',        Image,      self.image_cb, queue_size=1)
        self.pub = rospy.Publisher('/line/offset_heading', Float32MultiArray, queue_size=1)

        # Start Flask server
        flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False))
        flask_thread.daemon = True
        flask_thread.start()

        rospy.loginfo(f"[LineHeadingNode] Streaming video on http://localhost:5000/video_feed")
        rospy.loginfo(f"[LineHeadingNode] Waiting for CameraInfo at {self.cam_fps}Hz...")
        rospy.spin()

    def cam_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3,3)
            self.dist = np.array(msg.D)
            self.binning_x = msg.binning_x or 1
            self.binning_y = msg.binning_y or 1
            self.img_w = int(msg.width  / self.binning_x)
            self.img_h = int(msg.height / self.binning_y)
            self.roi = msg.roi
            rospy.loginfo(f"[LineHeadingNode] Calibrated: {self.img_w}x{self.img_h}, binning=({self.binning_x},{self.binning_y}), ROI={self.roi}")

    def image_cb(self, msg):
        stamp = msg.header.stamp or rospy.Time.now()
        if stamp - self.last_stamp < self.min_dt:
            return
        self.last_stamp = stamp
        if self.K is None:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

                # Undistort & crop
        img_undist = cv2.undistort(img, self.K, self.dist)
        if self.roi.height>0 and self.roi.width>0:
            x0,y0 = self.roi.x_offset, self.roi.y_offset
            img_undist = img_undist[y0:y0+self.roi.height, x0:x0+self.roi.width]
        frame = cv2.resize(img_undist, (self.img_w, self.img_h))

        # Threshold black regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_thresh = rospy.get_param('~black_thresh', 90)
        _, mask_black = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

                        # Plot two lines of dots at the leftmost and rightmost edges of the black mask
        h, w = mask_black.shape
        # Only process rows between 30% and 66% of the image height (ignore bottom third)
        for y in range(int(h*0.3), int(h*0.55), 10):  # every 10px from 30% downwards
            cols = np.where(mask_black[y] > 0)[0]
            if cols.size > 0:
                x_left = int(cols.min())
                x_right = int(cols.max())
                x_mid = (x_left + x_right) // 2
                # draw dots at both edges
                # cv2.circle(frame, (x_left, y), 3, (0, 255, 0), -1)
                cv2.circle(frame, (x_mid, y), 3, (0, 255, 0), -1)

        # Stream annotated frame and skip further processing
        self.publish_frame(frame)
        return


    def publish_frame(self, frame):
        global latest_frame, frame_lock
        with frame_lock:
            latest_frame = frame.copy()

if __name__=='__main__':
    try:
        LineHeadingNode()
    except rospy.ROSInterruptException:
        pass