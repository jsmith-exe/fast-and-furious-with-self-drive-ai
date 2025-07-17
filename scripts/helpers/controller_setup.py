import numpy as np
from controllers.PID import PID
from controllers.MPC import MPC
from typing import Union

class ControllerSetup:
    def __init__(self,
                 kp: float = 0,
                 kd: float = 0,
                 ki: float = 0,
                 integral_reset: Union[float, None] = None,
                 max_steer: float = 0,
                 n_states: int = 1,
                 n_controls: int = 1,
                 N: int = 0,
                 dt: float = 0,
                 w_y: float = 0,
                 w_delta: float = 0
                 ):

        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_reset = integral_reset
        self.max_steer = max_steer
        self.n_states = n_states
        self.n_controls = n_controls
        self.N = N
        self.dt = dt
        self.w_y = w_y
        self.w_delta = w_delta

    def system_dynamics(self):
        # Minimal lateral-integrator model:  ẏ = δ
        return lambda y, d: d

    def get_controller(self, controller: str) -> Union[PID, MPC, None]:
        if controller == "pid":
            return PID(Kp=self.kp, Ki=self.ki, Kd=self.ki, integral_reset=self.integral_reset, max_value=self.max_steer)
        elif controller == "mpc":
            return MPC(f=self.system_dynamics(),
                       n_states=self.n_states,
                       n_controls=self.n_controls,
                       N=self.N,
                       dt=self.dt,
                       Q=np.array([[self.w_y]]),
                       R=np.array([[self.w_delta]]),
                       u_bounds=(-self.max_steer, self.max_steer))
        else:
            print("Unknown controller")
            return None