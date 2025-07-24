import cv2
import numpy as np
from jetcam.csi_camera import CSICamera

# Reference area and distance (from calibration)
REFERENCE_AREA = 43456  # pixels
REFERENCE_DISTANCE = 0.20  # meters

# Initialize camera
cam = CSICamera(width=224, height=224, capture_width=1280, capture_height=720, capture_fps=30)
frame = cam.read()

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Simple threshold to segment main object
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area = w * h
    # Estimate distance using single reference point
    if area > 0:
        distance = REFERENCE_DISTANCE * np.sqrt(REFERENCE_AREA / area)
        print(f"Estimated distance: {distance:.3f} m (area={area} pixels)")
    else:
        print("Object area is zero.")
else:
    print("No object detected.")
