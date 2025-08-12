#!/usr/bin/env python
"""
ROS node /line/offset_yaw to detect black lane edges and compute lateral deviation & yaw relative to camera centerline, streaming MJPEG.
Turning streaming off saves processing power.
"""
# Turn streaming on and off
stream = False

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
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
        self.camera_height   = rospy.get_param('~camera_height', 0.3)   # camera height (m)
        self.cam_fps         = rospy.get_param('~cam_fps', 30)         # processing rate (Hz)
        self.black_thresh    = rospy.get_param('~black_thresh', 90)    # threshold for black mask

        # State
        self.K         = None
        self.dist      = None
        self.img_w     = None
        self.img_h     = None
        self.roi       = None
        self.last_stamp = rospy.Time(0)
        self.min_dt    = rospy.Duration(1.0 / self.cam_fps)

        # ROS interfaces
        self.pub = rospy.Publisher('/line/offset_yaw', Float32MultiArray, queue_size=1)
        rospy.Subscriber('/csi_cam_0/camera_info', CameraInfo, self.cam_info_cb)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.image_cb, queue_size=1)

        # Start MJPEG server
        if stream:
            flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False))
            flask_thread.daemon = True
            flask_thread.start()

            rospy.loginfo("[LineHeadingNode] MJPEG streaming on http://localhost:5000/video_feed")

        else:
            rospy.loginfo("[LineHeadingNode] Not Streaming")
        rospy.spin()

    def cam_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3,3)
            self.dist = np.array(msg.D)
            self.img_w = msg.width
            self.img_h = msg.height
            self.roi   = msg.roi

    def image_cb(self, msg):
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

        # Crop ROI if specified
        if self.roi and self.roi.height > 0 and self.roi.width > 0:
            x0, y0 = self.roi.x_offset, self.roi.y_offset
            frame = frame[y0:y0+self.roi.height, x0:x0+self.roi.width]

        # Threshold black lane
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_black = cv2.threshold(gray, self.black_thresh, 255, cv2.THRESH_BINARY_INV)

        # Sample midpoints in top 50%
        h, w = mask_black.shape
        mid_pts = []
        for y in range(int(h*0.3), int(h*0.55), 10):
            cols = np.where(mask_black[y] > 0)[0]
            if cols.size:
                mid_pts.append((int(np.mean(cols)), y))

        # Compute metrics if enough points
        if len(mid_pts) >= 2:
            mid_pts_sorted = sorted(mid_pts, key=lambda p: p[1])
            u_far, v_far   = mid_pts_sorted[0]
            u_near, v_near = mid_pts_sorted[-1]
            # Pixel lateral deviation from image center
            center_x   = w / 2.0
            lateral_px = u_near - center_x
            # Convert pixels to meters
            known_real_distance_metres = 0.1
            known_px = 192
            lateral_m         = lateral_px * ( known_real_distance_metres / known_px )
            fx = self.K[0,0]
            dx = u_far - u_near
            pixel_angle = np.arctan2(dx, fx)
            scale = (lateral_m / 1.8) / -0.04
            camera_pitch_tilt = np.deg2rad(6)
            
            yaw_rad = pixel_angle - (camera_pitch_tilt * scale)
            
            if stream:
                # Annotate & publish
                cv2.putText(frame, f"lat={lateral_m:.3f}m, yaw={np.rad2deg(yaw_rad):.2f}deg", 
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            self.pub.publish(Float32MultiArray(data=[lateral_m, yaw_rad]))

        # Draw midpoint dots
        for u, v in mid_pts:
            cv2.circle(frame, (u, v), 3, (0,255,0), -1)

        if stream:
            # Stream frame
            global latest_frame
            with frame_lock:
                latest_frame = frame.copy()

if __name__ == '__main__':
    try:
        LineHeadingNode()
    except rospy.ROSInterruptException:
        pass
