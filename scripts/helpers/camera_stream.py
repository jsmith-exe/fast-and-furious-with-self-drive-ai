from jetcam.csi_camera import CSICamera
from flask import Flask, Response
import cv2

# Initialize the Jetson CSI camera
camera = CSICamera(width=224, height=224, capture_fps=65)
camera.running = True

# Start Flask app
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.value
            if frame is None:
                continue
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            # Stream in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("📷 Starting JetCam stream at http://<your-ip>:5000/video_feed")
    app.run(host='0.0.0.0', port=5000)
