import time
from abc import ABC, abstractmethod
import casadi as ca
import numpy as np
from typing import Tuple

class ControlStrategy(ABC):
    @abstractmethod
    def compute_steering(self, state):
        '''Given current state, return steering command.'''
        pass

class MPC(ControlStrategy):
    '''
    Generic Model Predictive Controller.

    Parameters
    ----------
    f : callable
        Continuous‑time dynamics function f(x, u) -> ẋ or discrete dynamics x_next = f(x, u).
        Internally a forward Euler step with dt is applied if f gives ẋ.
    n_states : int
        Dimension of the state vector.
    n_controls : int
        Dimension of the control vector.
    N : int
        Prediction horizon.
    dt : float
        Sampling time [s].
    Q, R : ndarray
        State and control cost weight matrices (positive semi‑definite).
    x_bounds, u_bounds : tuple(ndarray, ndarray), optional
        Lower/upper bounds on states and controls.
    discrete_dynamics : bool
        If True `f` already returns x_{k+1}`.`
    '''
    def __init__(self,
                 f,
                 n_states,
                 n_controls,
                 N,
                 dt,
                 Q,
                 R,
                 x_bounds=None,
                 u_bounds=None,
                 discrete_dynamics=False,
                 solver='ipopt',
                 solver_opts=None):

        self.n_states = n_states
        self.n_controls = n_controls
        self.N = N
        self.dt = dt
        self.f = f
        self.discrete_dynamics = discrete_dynamics

        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(n_states, N + 1)
        self.U = self.opti.variable(n_controls, N)

        # Parameters
        self.x0 = self.opti.parameter(n_states)

        # Convert weighting matrices to CasADi DM
        Q_ca = ca.DM(Q) if isinstance(Q, (np.ndarray, list)) else Q
        R_ca = ca.DM(R) if isinstance(R, (np.ndarray, list)) else R

        # Build cost
        cost = 0
        for k in range(N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            cost += ca.mtimes([xk.T, Q_ca, xk]) + ca.mtimes([uk.T, R_ca, uk])
        self.opti.minimize(cost)

        # Dynamics constraints
        for k in range(N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            if discrete_dynamics:
                x_next = f(xk, uk)
            else:
                x_next = xk + f(xk, uk) * dt
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Bounds
        if u_bounds is not None:
            u_min, u_max = u_bounds
            self.opti.subject_to(self.opti.bounded(u_min, self.U, u_max))
        if x_bounds is not None:
            x_min, x_max = x_bounds
            self.opti.subject_to(self.opti.bounded(x_min, self.X, x_max))

        # Initial condition equality constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # Initial guess
        self.opti.set_initial(self.X, 0)
        self.opti.set_initial(self.U, 0)

        # Solver
        default_opts = {'print_time': False, 'ipopt': {'print_level': 0}}
        if solver_opts:
            # merge but keep keys nested
            default_opts['ipopt'].update(solver_opts.get('ipopt', {}))
            others = {k: v for k, v in solver_opts.items() if k != 'ipopt'}
            default_opts.update(others)
        self.opti.solver(solver, default_opts)

        # solution storage for warm start
        self._prev_sol = None

    def compute_steering(self, error: float) -> Tuple[float, float]:
        '''
        Solve MPC and return first control action and solve latency (ms).
        '''
        self.opti.set_value(self.x0, np.asarray(error).flatten())

        # warm start
        if self._prev_sol is not None:
            try:
                self.opti.set_initial(self.X, self._prev_sol.value(self.X))
                self.opti.set_initial(self.U, self._prev_sol.value(self.U))
            except RuntimeError:
                # previous solution infeasible; ignore
                pass

        tic = time.time()
        sol = self.opti.solve()
        latency_ms = (time.time() - tic) * 1000.0

        self._prev_sol = sol

        control = float(sol.value(self.U[0, 0]))
        return control, latency_ms
