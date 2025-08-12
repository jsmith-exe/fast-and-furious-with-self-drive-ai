#!/usr/bin/env python3
import io, time, threading
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, Response, render_template_string, redirect, jsonify
import rospy
from std_msgs.msg import Float32MultiArray

# ================== Config ==================
CAPTURE_SECONDS = 15
MAX_POINTS = 500
# ============================================

# Buffers
time_axis = deque(maxlen=MAX_POINTS)
lateral_errors = deque(maxlen=MAX_POINTS)
heading_errors_deg = deque(maxlen=MAX_POINTS)

# State
buf_lock = threading.Lock()
capturing = False
frozen = False
capture_start_mono = None          # <-- monotonic start
start_time_ros = None
cached_lat_png = None
cached_yaw_png = None
finalize_timer = None              # <-- one-shot timer handle

def reset_capture():
    """Prepare for a new capture and start the one-shot finalize timer."""
    global capturing, frozen, capture_start_mono, start_time_ros
    global cached_lat_png, cached_yaw_png, finalize_timer
    with buf_lock:
        time_axis.clear()
        lateral_errors.clear()
        heading_errors_deg.clear()
        cached_lat_png = None
        cached_yaw_png = None
        start_time_ros = None
        capture_start_mono = time.monotonic()
        capturing = True
        frozen = False

    # cancel any previous timer
    if finalize_timer is not None:
        try:
            finalize_timer.cancel()
        except Exception:
            pass
    # start new one-shot timer that *always* fires after CAPTURE_SECONDS
    def _finalize_safe():
        try:
            finalize_capture()
        except Exception as e:
            rospy.logerr(f"Finalize failed: {e}")
    t = threading.Timer(CAPTURE_SECONDS, _finalize_safe)
    t.daemon = True
    t.start()

    # store handle
    globals()['finalize_timer'] = t

def finalize_capture():
    """Render PNGs once and cache them (idempotent)."""
    global cached_lat_png, cached_yaw_png, capturing, frozen
    # if already frozen, no-op
    if frozen:
        return

    with buf_lock:
        x = list(time_axis)
        y_lat = list(lateral_errors)
        y_yaw = list(heading_errors_deg)

    # Lateral plot
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.set_title(f"Lateral Error (m) — Final {int(CAPTURE_SECONDS)}s")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Lateral (m)")
    ax1.grid(True, alpha=0.3); ax1.set_ylim(-0.3, 0.3)  # tweak as needed
    if x:
        ax1.plot(x, y_lat)
        ax1.set_xlim(max(0, x[-1]-CAPTURE_SECONDS), x[-1] + 0.5)
    else:
        ax1.text(0.5, 0.5, "No data captured", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=12)
    buf1 = io.BytesIO(); fig1.tight_layout(); fig1.savefig(buf1, format="png"); plt.close(fig1); buf1.seek(0)
    cached_lat_png = buf1.read()

    # Yaw plot
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.set_title(f"Heading Error (deg) — Final {int(CAPTURE_SECONDS)}s")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Yaw (deg)")
    ax2.grid(True, alpha=0.3); ax2.set_ylim(-45, 45)
    if x:
        ax2.plot(x, y_yaw)
        ax2.set_xlim(max(0, x[-1]-CAPTURE_SECONDS), x[-1] + 0.5)
    else:
        ax2.text(0.5, 0.5, "No data captured", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12)
    buf2 = io.BytesIO(); fig2.tight_layout(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    cached_yaw_png = buf2.read()

    with buf_lock:
        capturing = False
        frozen = True

# ROS callback — only buffers while capturing
def line_cb(msg: Float32MultiArray):
    global start_time_ros
    if not capturing or frozen:
        return
    t_ros = rospy.get_time()
    if start_time_ros is None:
        start_time_ros = t_ros
    t = max(0.0, t_ros - start_time_ros)

    lat = float(msg.data[0]) if len(msg.data) > 0 else 0.0
    hdg_rad = float(msg.data[1]) if len(msg.data) > 1 else 0.0

    with buf_lock:
        time_axis.append(t)
        lateral_errors.append(lat)
        heading_errors_deg.append(np.rad2deg(hdg_rad))

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{{ cap }}s Capture — Lateral & Yaw</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
      .wrap { max-width: 980px; margin: 24px auto; padding: 0 12px; }
      h1 { font-size: 18px; font-weight: 600; }
      .row { display: grid; grid-template-columns: 1fr; gap: 16px; }
      img { width: 100%; height: auto; border-radius: 12px; background:#000; }
      .controls { margin: 8px 0 16px; opacity:.85; font-size:14px; display:flex; gap:12px; align-items:center; }
      button { background:#2d6cdf; color:#fff; border:none; border-radius:10px; padding:8px 12px; cursor:pointer; }
      .muted { opacity:.8; }
      @media (min-width: 900px) { .row { grid-template-columns: 1fr 1fr; } }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>/line/offset_yaw — Final graphs after {{ cap }}s</h1>
      <div class="controls">
        <button onclick="startCapture()">Start {{ cap }}s capture</button>
        <span id="status" class="muted">Idle</span>
      </div>
      <div id="plots" class="row"></div>
    </div>
    <script>
      let timer = null;
      async function poll() {
        const r = await fetch('/status'); const s = await r.json();
        const st = document.getElementById('status');
        if (s.capturing) {
          st.textContent = `Capturing… ${s.seconds_elapsed.toFixed(1)}s / ${s.capture_seconds}s  (points: ${s.points})`;
        } else if (s.frozen) {
          st.textContent = 'Done. Final graphs below.';
          const div = document.getElementById('plots');
          div.innerHTML = `
            <img src="/lat.png?ts=${Date.now()}"/>
            <img src="/yaw.png?ts=${Date.now()}"/>
          `;
          if (timer) { clearInterval(timer); timer = null; }
        } else {
          st.textContent = 'Idle — click Start {{ cap }}s capture';
        }
      }
      function startCapture() {
        fetch('/start').then(_ => {
          document.getElementById('plots').innerHTML = '';
          if (timer) clearInterval(timer);
          timer = setInterval(poll, 500);
          poll();
        });
      }
      poll();
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, cap=int(CAPTURE_SECONDS))

@app.route("/start")
def start():
    reset_capture()
    return redirect("/")

@app.route("/status")
def status():
    with buf_lock:
        secs = time.monotonic() - capture_start_mono if capturing and capture_start_mono else 0.0
        return jsonify({
            "capturing": capturing,
            "frozen": frozen,
            "capture_seconds": CAPTURE_SECONDS,
            "seconds_elapsed": max(0.0, float(secs)),
            "points": len(time_axis),
        })

@app.route("/lat.png")
def lat_png():
    if not frozen or cached_lat_png is None:
        return Response(b"", status=404)
    return Response(cached_lat_png, mimetype="image/png")

@app.route("/yaw.png")
def yaw_png():
    if not frozen or cached_yaw_png is None:
        return Response(b"", status=404)
    return Response(cached_yaw_png, mimetype="image/png")

def ros_thread():
    rospy.init_node("error_plot_server_capture", anonymous=True, disable_signals=True)
    rospy.Subscriber("/line/offset_yaw", Float32MultiArray, line_cb)
    rospy.spin()

if __name__ == "__main__":
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=6000, debug=False, threaded=True)
