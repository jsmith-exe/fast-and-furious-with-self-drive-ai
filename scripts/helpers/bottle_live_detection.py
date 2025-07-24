import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
import collections
from flask import Flask, Response


# === HSV Color Range for Bottle Detection ===
# TUNE HERE: Set the lower and upper HSV bounds for your bottle color
# Example: For a maroon bottle, start with these and adjust as needed
LOWER_H = 0   # Hue lower bound (0-180)
UPPER_H = 100  # Hue upper bound (0-180)
LOWER_S = 60  # Saturation lower bound (0-255)
UPPER_S = 255 # Saturation upper bound (0-255)
LOWER_V = 60  # Value lower bound (0-255)
UPPER_V = 255 # Value upper bound (0-255)

LOWER_HSV = np.array([LOWER_H, LOWER_S, LOWER_V])
UPPER_HSV = np.array([UPPER_H, UPPER_S, UPPER_V])

print(f"Tuning HSV: LOWER_HSV={LOWER_HSV}, UPPER_HSV={UPPER_HSV}")


# Reference calibration
REFERENCE_AREA = 43456  # pixels
REFERENCE_DISTANCE = 0.20  # meters


bottle_camera = CSICamera(width=224, height=224, capture_fps=30)
# camera.running = True  # Removed for direct frame reading
bbox_history = collections.deque(maxlen=5)
app = Flask(__name__)

bottle_camera.running = False
cv2.destroyAllWindows()

def gen_frames():
    # Print HSV bounds for live tuning
    print(f"Current HSV bounds: LOWER_HSV={LOWER_HSV}, UPPER_HSV={UPPER_HSV}")
    while True:
        frame = bottle_camera.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7,7), 0)
        # Use single HSV range for easier tweaking
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        min_area = 500
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_area:
                x, y, w, h = cv2.boundingRect(largest)
                bbox_history.append((x, y, w, h))
                if len(bbox_history) > 1:
                    x = int(np.mean([b[0] for b in bbox_history]))
                    y = int(np.mean([b[1] for b in bbox_history]))
                    w = int(np.mean([b[2] for b in bbox_history]))
                    h = int(np.mean([b[3] for b in bbox_history]))
                if w > 0:
                    # Use width for distance estimation
                    REFERENCE_WIDTH = 38  # Set this to the width (pixels) at REFERENCE_DISTANCE
                    distance = REFERENCE_DISTANCE * (REFERENCE_WIDTH / w)
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(annotated, f'Dist: {distance:.2f}m', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
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
