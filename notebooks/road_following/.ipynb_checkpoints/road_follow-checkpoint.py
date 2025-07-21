import sys
sys.path.insert(0, '/jetracer/jetracer')
sys.path.insert(1, '/jetcam/jetcam')
import torch
from torch2trt import TRTModule
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
from utils import preprocess
from jetcam.utils import bgr8_to_jpeg

def main():
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('model_08-06_trt.pth'))
    car = NvidiaRacecar()
    camera = CSICamera(width=224, height=224, capture_fps=65)
    car.steering_offset = 0
    car.steering_gain = -1
    car.throttle = 0.2
    while(true):
        image = preprocess(new_image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        x = float(output[0])
        car.steering = x
