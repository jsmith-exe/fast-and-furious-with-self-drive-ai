# scripts/collision_avoidance/object_detection_follow.py
import cv2, collections, threading, numpy as np
from jetcam.csi_camera import CSICamera
from flask import Flask, Response, jsonify

class ObjectLiveDetector:
    """
    Detects orange blobs and streams MJPEG/JSON.
    Shares a single CSI camera and a lock to serialize reads.
    """

    def __init__(
        self,
        camera=None,
        width=224,
        height=224,
        capture_fps=65,
        reference_distance=0.20,
        reference_width=45,
        lower_hsv=np.array([10, 40, 50]),
        upper_hsv=np.array([40, 140, 150]),
        alpha=0.2,
        min_area=500,
        camera_lock: threading.Lock = None,
    ):
        self.camera = camera
        self.width, self.height = width, height
        self.capture_fps = capture_fps
        self.ref_dist, self.ref_width = reference_distance, reference_width
        self.lower_hsv, self.upper_hsv = lower_hsv, upper_hsv
        self.alpha = alpha
        self.min_area = min_area
        self.bbox_history = collections.deque(maxlen=5)
        self.filtered_distance = None
        self.latest_frame = None
        self.latest_bbox = None
        self.latest_distance = None
        self.lock = threading.Lock()
        self.camera_lock = camera_lock
        self.running = False
        self._thread = None

    def _ensure_camera(self):
        if self.camera is None:
            self.camera = CSICamera(width=self.width, height=self.height,
                                     capture_fps=self.capture_fps)

    def _capture_loop(self):
        self._ensure_camera()
        while self.running:
            # serialize camera read
            if self.camera_lock:
                with self.camera_lock:
                    frame = self.camera.read()
            else:
                frame = self.camera.read()
            if frame is None:
                continue

            # blur + morphology
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)
            mask = cv2.inRange(hsv_blur, self.lower_hsv, self.upper_hsv)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # find contours above threshold
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            bbox, distance = None, None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > self.min_area:
                    x, y, w_box, h_box = cv2.boundingRect(largest)
                    self.bbox_history.append((x, y, w_box, h_box))
                    # smooth bounding box
                    if len(self.bbox_history) > 1:
                        x = int(np.mean([b[0] for b in self.bbox_history]))
                        y = int(np.mean([b[1] for b in self.bbox_history]))
                        w_box = int(np.mean([b[2] for b in self.bbox_history]))
                        h_box = int(np.mean([b[3] for b in self.bbox_history]))
                    # compute distance
                    if w_box > 0:
                        raw_dist = self.ref_dist * (self.ref_width / w_box)
                        if self.filtered_distance is None:
                            self.filtered_distance = raw_dist
                        else:
                            self.filtered_distance = (
                                self.alpha * raw_dist +
                                (1 - self.alpha) * self.filtered_distance
                            )
                        distance = self.filtered_distance
                        bbox = (x, y, w_box, h_box)

            with self.lock:
                self.latest_frame = frame
                self.latest_bbox = bbox
                self.latest_distance = distance

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def get_latest(self):
        with self.lock:
            return self.latest_distance, self.latest_bbox, self.latest_frame

    def gen_frames(self):
        while True:
            dist, bbox, frame = self.get_latest()
            if frame is None:
                continue
            annotated = frame.copy()
            if bbox and dist is not None:
                x, y, w_box, h_box = bbox
                cv2.rectangle(annotated, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                cv2.putText(annotated, f'Dist: {dist:.2f}m', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            ret, buf = cv2.imencode('.jpg', annotated)
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def create_flask_app(detector: ObjectLiveDetector) -> Flask:
    app = Flask(__name__)
    @app.route('/video_feed')
    def video_feed():
        return Response(detector.gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/evasion_data')
    def evasion_data():
        dist, bbox, _ = detector.get_latest()
        return jsonify({'distance': dist, 'bbox': bbox})
    return app