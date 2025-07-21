import cv2
import torch
from torchvision.models import resnet18
from torch2trt import TRTModule
from utils import preprocess
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
import time

# Load the TensorRT-optimized road following model
print("Loading TensorRT model…")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('trained_models/updated_model_trt.pth'))
model_trt.eval()
print("✅ Model loaded.")

# Create the racecar class
print("Initializing racecar…")
car = NvidiaRacecar()
car.steering_gain = -1.0
car.throttle = 0.0
print("✅ Racecar ready (steering_gain={}, throttle={}).".format(car.steering_gain, car.throttle))

# Create the camera class
print("Starting camera…")
camera = CSICamera(width=224, height=224, capture_fps=65)
print("✅ Camera streaming at {}×{} @ {}fps.".format(224, 224, 65))

STEERING_GAIN = 2.0
STEERING_BIAS = 0.00

# For limiting print output frequency
last_print_time = time.time()
frame_count = 0

kp = 2
kd = 0.5
last_x = 0

print("Entering control loop…")
# Start prediction-control loop
while True:
    frame_count += 1
    print("Frame #{}: Capturing…".format(frame_count), end=' ')
    image = camera.read()  # Capture frame from camera
    print("Done.")

    print("Frame #{}: Preprocessing…".format(frame_count), end=' ')
    tensor = preprocess(image).half()
    print("Done.")

    print("Frame #{}: Running inference…".format(frame_count), end=' ')
    with torch.no_grad():
        output = model_trt(tensor).detach().cpu().numpy().flatten()
    x = float(output[0])  # Predicted normalized x coordinate
    print("Done. x_norm={:.3f}".format(x))

    # Convert prediction to actual steering command
    steering_cmd = x * STEERING_GAIN + STEERING_BIAS
    steering_pd =  (x * kp + (x - last_x) * kd)
    car.steering = steering_pd
    print("Frame #{}: Applied steering {:.3f}".format(frame_count, steering_cmd))

    # Print output every 2 seconds for debugging
    current_time = time.time()
    if current_time - last_print_time >= 2.0:
        print("-- Status @ {:.1f}s -- Apex x: {:.3f}, Steering: {:.3f}".format(
            current_time, x, steering_cmd))
        last_print_time = current_time
