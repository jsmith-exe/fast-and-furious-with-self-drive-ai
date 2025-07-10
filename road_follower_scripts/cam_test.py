from jetcam.csi_camera import CSICamera
cam = CSICamera(width=224, height=224,
                capture_width=1280, capture_height=720,
                capture_fps=30)
frame = cam.read()      # should be an ndarray
print(frame.shape)