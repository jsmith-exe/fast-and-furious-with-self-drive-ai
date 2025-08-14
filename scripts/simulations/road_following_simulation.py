from controllers_sim.PID import PID
from controllers_sim.MPC import MPC
from controller_setup import ControllerSetup
from sim_class import LineFollowerSim
import numpy as np

# === Select Sim Modes ---------------------------------------------------------
''' controller types: 
    - pid : use a Proportional Integral Differential Controller
    - mpc : use Model Predictive Controller              
'''
''' detection_methods types:
    - cte : use the exact Cross Track Error, which is the perpendicular distance between the car and the track
    - camera : use the simulation camera value of track distance          
'''
controller = "pid"
detection_method = "cte"

# === Reference path === -------------------------------------------------------
starting_offset = 0 # (m)
course_length: int = 1200
course_amp  = 0.5
course_freq = 0.7

# === MPC parameters === ---------------------------------------
N  = 40          # prediction horizon
dt = 0.05        # [s] integration step
v  = 3         # [m/s] constant forward speed
max_steer = np.deg2rad(40)  # [rad] hard steering limit
max_cte = None # max cte

# Cost weights
w_y     = 5.0   # weight on lateral error (y)
w_delta = 0.5    # weight on steering usage (delta)

# === PID parameters === ---------------------------------------
KP = 2.0
KI = 0.1
KD = 0.2
integral_reset = None

def run_sim(controller: PID | MPC) -> None:
    sim = LineFollowerSim(
                 controller=controller,
                 course_length = course_length,
                 car_velocity= v,
                 car_starting_offset = starting_offset,
                 course_amplitude = course_amp,
                 course_freq = course_freq,
                 line_detection= detection_method )
    sim.run()

def main() -> None:
    controller_class = ControllerSetup(
        kp=KP,
        ki=KI,
        kd=KD,
        integral_reset=integral_reset,
        max_steer=max_steer,
        N=N,
        dt=dt,
        w_y=w_y,
        w_delta=w_delta,
    )
    ctrl = controller_class.get_controller(controller=controller)
    run_sim(controller=ctrl)

if __name__ == '__main__':
    main()



