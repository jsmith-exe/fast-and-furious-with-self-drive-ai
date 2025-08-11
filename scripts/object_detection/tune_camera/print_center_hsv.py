'''
Used to determine the colour of the object you are trying to detect using its HSV value.
This script will tell you the HSV value of any object in the centre pixel frame (camera lens).
'''

import cv2
import numpy as np
from jetcam.csi_camera import CSICamera

# Initialize camera
camera = CSICamera(width=224, height=224, capture_fps=30)
camera.running = False  # Must be False for direct 

try:
    while True:
        frame = camera.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        center_x = hsv.shape[1] // 2
        center_y = hsv.shape[0] // 2
        center_pixel_hsv = hsv[center_y, center_x]
        print(f"HSV value at center ({center_x}, {center_y}): {center_pixel_hsv}")
except KeyboardInterrupt:
    print("Stopped.")
finally:
    camera.running = False
    cv2.destroyAllWindows()
