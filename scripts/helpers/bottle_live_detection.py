import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
import collections
from flask import Flask, Response
import threading


# === HSV Color Range for Bottle Detection ===
# TUNE HERE: Set the lower and upper HSV bounds for your bottle color
# Example: For a maroon bottle, start with these and adjust as needed
LOWER_H = 10  # Hue lower bound (0-180)
UPPER_H = 40  # Hue upper bound (0-180)
LOWER_S = 40  # Saturation lower bound (0-255)
UPPER_S = 140 # Saturation upper bound (0-255)
LOWER_V = 50  # Value lower bound (0-255)
UPPER_V = 150 # Value upper bound (0-255)

LOWER_HSV = np.array([LOWER_H, LOWER_S, LOWER_V])
UPPER_HSV = np.array([UPPER_H, UPPER_S, UPPER_V])

print(f"Tuning HSV: LOWER_HSV={LOWER_HSV}, UPPER_HSV={UPPER_HSV}")

REFERENCE_DISTANCE = 0.20  # meters
REFERENCE_WIDTH = 45  # Set this to the width (pixels) at REFERENCE_DISTANCE


bottle_camera = CSICamera(width=224, height=224, capture_fps=65)
# camera.running = True  # Removed for direct frame reading
bbox_history = collections.deque(maxlen=5)


app = Flask(__name__)

bottle_camera.running = False
cv2.destroyAllWindows()

shutdown_flag = threading.Event()

def gen_frames():
    print(f"Current HSV bounds: LOWER_HSV={LOWER_HSV}, UPPER_HSV={UPPER_HSV}")
    filtered_distance = None
    alpha = 0.2  # Smoothing factor for low-pass filter (0 < alpha <= 1)
    while True:
        frame = bottle_camera.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_blur = cv2.GaussianBlur(hsv, (7,7), 0)
        mask = cv2.inRange(hsv_blur, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        min_area = 500
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_area:
                x, y, w_box, h_box = cv2.boundingRect(largest)
                bbox_history.append((x, y, w_box, h_box))
                if len(bbox_history) > 1:
                    x = int(np.mean([b[0] for b in bbox_history]))
                    y = int(np.mean([b[1] for b in bbox_history]))
                    w_box = int(np.mean([b[2] for b in bbox_history]))
                    h_box = int(np.mean([b[3] for b in bbox_history]))
                if w_box > 0:
                    distance = REFERENCE_DISTANCE * (REFERENCE_WIDTH / w_box)
                    if filtered_distance is None:
                        filtered_distance = distance
                    else:
                        filtered_distance = alpha * distance + (1 - alpha) * filtered_distance
                    print(f"Detected object at (x={x}, y={y}, w={w_box}, h={h_box}), Distance: {filtered_distance:.2f}m (raw: {distance:.2f}m)")
                    cv2.rectangle(annotated, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                    cv2.putText(annotated, f'Dist: {filtered_distance:.2f}m', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
bottle_camera.running = False
cv2.destroyAllWindows()
