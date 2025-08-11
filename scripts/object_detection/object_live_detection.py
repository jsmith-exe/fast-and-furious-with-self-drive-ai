import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
import collections
from flask import Flask, Response, jsonify
import threading


class ObjectLiveDetector:
    def __init__(self, width=224, height=224, capture_fps=65, reference_distance=0.20, reference_width=45,
                 lower_hsv=np.array([10, 40, 50]), upper_hsv=np.array([40, 140, 150])):
        self.camera = None
        self.width = width
        self.height = height
        self.capture_fps = capture_fps
        self.bbox_history = collections.deque(maxlen=5)
        self.reference_distance = reference_distance
        self.reference_width = reference_width
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.filtered_distance = None
        self.alpha = 0.2  # Smoothing factor for low-pass filter
        self.latest_frame = None
        self.latest_distance = None
        self.latest_bbox = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)

    def _ensure_camera(self):
        if self.camera is None:
            try:
                self.camera = CSICamera(width=self.width, height=self.height, capture_fps=self.capture_fps)
                self.camera.running = True
            except Exception as e:
                print(f"Camera initialization failed: {e}")
                self.camera = None
                return

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()

    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.running = False

    def _update_loop(self):
        self._ensure_camera()
        while self.running and self.camera is not None:
            try:
                frame = self.camera.value
            except Exception as e:
                print(f"Camera read failed: {e}")
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_blur = cv2.GaussianBlur(hsv, (7,7), 0)
            mask = cv2.inRange(hsv_blur, self.lower_hsv, self.upper_hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 500
            bbox = None
            distance = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > min_area:
                    x, y, w_box, h_box = cv2.boundingRect(largest)
                    self.bbox_history.append((x, y, w_box, h_box))
                    if len(self.bbox_history) > 1:
                        x = int(np.mean([b[0] for b in self.bbox_history]))
                        y = int(np.mean([b[1] for b in self.bbox_history]))
                        w_box = int(np.mean([b[2] for b in self.bbox_history]))
                        h_box = int(np.mean([b[3] for b in self.bbox_history]))
                    if w_box > 0:
                        distance = self.reference_distance * (self.reference_width / w_box)
                        if self.filtered_distance is None:
                            self.filtered_distance = distance
                        else:
                            self.filtered_distance = self.alpha * distance + (1 - self.alpha) * self.filtered_distance
                        bbox = (x, y, w_box, h_box)
            with self.lock:
                self.latest_frame = frame
                self.latest_distance = self.filtered_distance if distance is not None else None
                self.latest_bbox = bbox

    def get_latest(self):
        with self.lock:
            return self.latest_distance, self.latest_bbox, self.latest_frame

    def gen_frames(self):
        while True:
            distance, bbox, frame = self.get_latest()
            if frame is None:
                continue
            annotated = frame.copy()
            if bbox:
                x, y, w_box, h_box = bbox
                cv2.rectangle(annotated, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                cv2.putText(annotated, f'Dist: {distance:.2f}m', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask app for streaming
app = Flask(__name__)



if __name__ == '__main__':
    detector = ObjectLiveDetector()
    detector.start()

    
    @app.route('/video_feed')
    def video_feed():
        return Response(detector.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/distance')
    def distance():
        dist, bbox, _ = detector.get_latest()
        if dist is not None:
            return {'distance': dist, 'bbox': bbox}
        else:
            return {'distance': None, 'bbox': None}

    @app.route('/evasion_data')
    def evasion_data():
        dist, bbox, frame = detector.get_latest()
        result = {
            'distance': dist,
            'bbox': bbox
        }
        return jsonify(result)

    app.run(host='0.0.0.0', port=5000)
    detector.stop()
    cv2.destroyAllWindows()
