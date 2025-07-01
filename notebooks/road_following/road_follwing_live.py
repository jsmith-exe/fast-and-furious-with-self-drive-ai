import torch
from torch2trt import TRTModule
import numpy as np
from collections import deque
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
import cv2
import ipywidgets
import threading


model_trt = TRTModule()
model_trt.load_state_dict(torch.load('trained_models/updated_model_trt.pth'))

prev_x = deque(maxlen=5)

car = NvidiaRacecar()

camera = CSICamera(width=224, height=224, capture_fps=65)

state_widget = ipywidgets.ToggleButtons(options=['On', 'Off'], description='Camera', value='On')
prediction_widget = ipywidgets.Image(format='jpeg', width=camera.width, height=camera.height)

live_execution_widget = ipywidgets.VBox([
    prediction_widget,
    state_widget
])

def road_confidence(cv_image):
    """Calculate confidence based on road color detection"""
    colour_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 60, 60])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(colour_img, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    height, width = mask.shape
    region = mask[int(height * 0.6):, int(width * 0.3):int(width * 0.7)]
    green_pixels = cv2.countNonZero(region)
    area = region.shape[0] * region.shape[1]
    return green_pixels / area if area > 0 else 0.0

def ml_confidence():
    """Calculate confidence based on steering stability"""
    if len(prev_x) < 2:  # Need at least 2 values for std
        return 1.0
    return 1.0 - np.var(list(prev_x)[-5:])