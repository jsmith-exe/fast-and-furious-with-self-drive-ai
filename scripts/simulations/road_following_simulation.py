from controllers.PID import PID
from controllers.MPC import MPC
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
controller = "mpc"
detection_method = "camera"

# === Reference path === -------------------------------------------------------
starting_offset = 0.50 # (m)
course_length: int = 1600
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

def system_dynamics():
    # Minimal lateral-integrator model:  ẏ = δ
    return lambda y, d: d

def get_controller() -> PID | MPC | None:
    if controller == "pid":
        return PID(Kp=KP, Ki=KI, Kd=KD, integral_reset=integral_reset, max_value=max_steer)
    elif controller == "mpc":
        return MPC(f=system_dynamics(),
              n_states=1,
              n_controls=1,
              N=N,
              dt=dt,
              Q=np.array([[w_y]]),
              R=np.array([[w_delta]]),
              #x_bounds=(-max_cte, max_cte),
              u_bounds=(-max_steer, max_steer))
    else:
        print("Unknown controller")
        return None

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
    ctrl = get_controller()
    run_sim(controller=ctrl)

if __name__ == '__main__':
    main()



