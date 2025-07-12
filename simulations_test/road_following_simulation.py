"""
line_follower_mpc_simulation.py
Run with:  python line_follower_mpc_simulation.py
"""
import os, sys, time, turtle, numpy as np
from typing import Tuple, Union

# --- Make sure Tcl/Tk is found (needed for turtle on some installs) ---
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY']  = os.path.join(base_prefix, 'tcl', 'tk8.6')

from controllers.MPC import MPC          # << our new reusable controller
from controllers.PID import PID          # << legacy option (if you want to compare)

# === Reference path === -------------------------------------------------------
course_amp  = 0.5
course_freq = 0.7
def course_function(x):
    "Desired y position for a given x (simple sine wave)."
    return course_amp * np.sin(course_freq * x)

# === Choose your controller here =============================================
USE_PID: bool = 0   # flip to False to try the MPC

# === MPC parameters === ---------------------------------------
N  = 40          # prediction horizon
dt = 0.05        # [s] integration step
v  = 0.8         # [m/s] constant forward speed
max_steer = np.pi/4  # [rad] hard steering limit

# Cost weights
w_y     = 5.0   # weight on lateral error (y)
w_delta = 0.5    # weight on steering usage (delta)

# === PID parameters === ---------------------------------------
KP = 10.0
KI = 1.0
KD = 0.15
INTEGRAL_RESET = 0.0
DELAY = 0.0

# Minimal lateral-integrator model:  ẏ = δ
dyn = lambda y, d: d
mpc = MPC(f=dyn,
          n_states=1,
          n_controls=1,
          N=N,
          dt=dt,
          Q=np.array([[w_y]]),
          R=np.array([[w_delta]]),
          u_bounds=(-max_steer, max_steer))

pid = PID(Kp=KP, Ki=KI, Kd=KD, integral_reset=INTEGRAL_RESET, delay=DELAY)  # tweak to taste

controller = pid if USE_PID else mpc

# Helper so the main loop looks identical for either controller
def get_command(ctrl: Union[PID, MPC], cte: float) -> Tuple[float, float]:
    """
        Compute a steering command from either a PID or MPC controller.

        Parameters
        ----------
        ctrl : PID | MPC
            An instance of your PID or MPC controller class.
        cte : float
            Cross-track error (metres).

        Returns
        -------
        Tuple[float, float]
            steer   – steering angle in **radians**
            latency – solver/runtime latency in **milliseconds**
                      (0 ms when using a PID).
        """
    if isinstance(ctrl, PID):
        steer = ctrl.compute_steering(-cte)   # scalar in, scalar out
        latency = 0.0
    else:  # MPC
        steer, latency = ctrl.compute_steering([cte])  # list/array state
    return steer, latency

# === Turtle visualisation (unchanged) ========================================
screen = turtle.Screen();  screen.setup(800, 600);  screen.title('Line follower')

# Draw the reference path once
path = turtle.Turtle(visible=False);  path.penup()
path.goto(-400, course_function(-4.0)*100);  path.pendown()
for px in range(-400, 401):
    xm = px / 100.0
    path.goto(px, course_function(xm)*100)

car = turtle.Turtle();  car.shape('arrow');  car.color('red');  car.penup()
car.goto(-400, course_function(-4.0)*100);  car.setheading(0);  car.pendown()

# === Main loop ===============================================================
x = -4.0                    # starting x-position (m)
for step in range(200):
    y_real = car.ycor()/100.0                  # turtle y (m)
    cte    = y_real - course_function(x)       # cross-track error

    # --- get steering command from the chosen controller ------------------
    steer, latency = get_command(controller, cte)

    steer = np.clip(steer, -max_steer, max_steer)

    # Log & visualise
    print(f"step {step:3d} | CTE={cte:+.3f} m | steer={np.rad2deg(steer):+.1f}°"
          f" | solver latency={latency:.1f} ms")


    car.setheading(np.rad2deg(steer))
    car.forward(v * 100 * dt)   # 100 px ≈ 1 m
    x += v * dt
    #time.sleep(dt)

turtle.done()
