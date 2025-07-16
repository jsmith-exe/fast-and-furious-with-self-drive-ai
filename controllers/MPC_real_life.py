import time
from abc import ABC, abstractmethod
import casadi as ca
import numpy as np
from typing import Tuple, Union, Sequence

class ControlStrategy(ABC):
    @abstractmethod
    def compute_steering(self, state):
        """Given current state, return steering command."""
        ...


class MPC(ControlStrategy):
    """
    Generic Model Predictive Controller.

    Parameters
    ----------
    f : callable
        Continuous‑time dynamics f(x, u) → ẋ  *or* discrete x⁺ = f(x, u).
    n_states, n_controls : int
        State / control dimensions.
    N : int
        Prediction horizon (control intervals).
    dt : float
        Sampling time [s].
    Q, R : ndarray | DM
        State- and control-weight matrices (PSD).
    x_bounds, u_bounds : tuple(ndarray, ndarray), optional
        (min, max) bounds on state and control.
    discrete_dynamics : bool
        If True, `f` already returns the next state.
    solver : str
        NLP backend name, default “ipopt”; will fall back to “sqpmethod” if
        the requested plugin is absent in the current CasADi build.
    solver_opts : dict
        Additional options forwarded to `opti.solver`.
    """

    def __init__(
        self,
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
        solver="ipopt",
        solver_opts=None,
    ):

        # ───────────── bookkeeping ──────────────
        self.n_states = n_states
        self.n_controls = n_controls
        self.N = N
        self.dt = dt
        self.f = f
        self.discrete_dynamics = discrete_dynamics

        self.opti = ca.Opti()

        # ───────── decision variables ───────────
        self.X = self.opti.variable(n_states, N + 1)
        self.U = self.opti.variable(n_controls, N)

        # ───────── parameter (current state) ────
        self.x0 = self.opti.parameter(n_states)

        # ───────── cost function ────────────────
        Q_ca = ca.DM(Q) if isinstance(Q, (np.ndarray, list)) else Q
        R_ca = ca.DM(R) if isinstance(R, (np.ndarray, list)) else R

        cost = 0
        for k in range(N):
            xk, uk = self.X[:, k], self.U[:, k]
            cost += ca.mtimes([xk.T, Q_ca, xk]) + ca.mtimes([uk.T, R_ca, uk])
        self.opti.minimize(cost)

        # ───────── dynamics constraints ─────────
        for k in range(N):
            xk, uk = self.X[:, k], self.U[:, k]
            if discrete_dynamics:
                x_next = f(xk, uk)
            else:
                x_next = xk + f(xk, uk) * dt
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # ───────── bounds ───────────────────────
        if u_bounds is not None:
            u_min, u_max = u_bounds
            self.opti.subject_to(self.opti.bounded(u_min, self.U, u_max))
        if x_bounds is not None:
            x_min, x_max = x_bounds
            self.opti.subject_to(self.opti.bounded(x_min, self.X, x_max))

        # initial condition equality & guess
        self.opti.subject_to(self.X[:, 0] == self.x0)
        self.opti.set_initial(self.X, 0)
        self.opti.set_initial(self.U, 0)

        # ───────── solver selection ─────────────
        try:
            available = set(ca.plugin_list("nlpsol"))  # CasADi ≥ 3.6
        except AttributeError:
            available = {"sqpmethod", "qrqp"}         # CasADi 3.5.x

        if solver not in available:
            print(f"[MPC] NLP solver '{solver}' not found; falling back to 'sqpmethod'.")
            solver = "sqpmethod"

        opts = {"print_time": False}

        if solver == "ipopt":
            opts["ipopt"] = {"print_level": 0}

        if solver == "sqpmethod":
            opts.update({
                "qpsol": "qrqp",  # use built‑in QP solver
                "max_iter": 15,
                "tol_pr": 1e-6,
                "tol_du": 1e-6,
            })

        # merge user overrides
        if solver_opts:
            if solver == "ipopt" and "ipopt" in solver_opts:
                opts["ipopt"].update(solver_opts["ipopt"])
            opts.update({k: v for k, v in solver_opts.items() if k != "ipopt"})

        self.opti.solver(solver, opts)

        # warm‑start buffer
        self._prev_sol = None

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────
    def compute_steering(self, error: Union[float, Sequence[float]]) -> Tuple[float, float]:
        """Return steering (rad) and solver latency in ms."""
        self.opti.set_value(self.x0, np.asarray(error).flatten())

        # warm start from previous solution
        if self._prev_sol is not None:
            try:
                self.opti.set_initial(self.X, self._prev_sol.value(self.X))
                self.opti.set_initial(self.U, self._prev_sol.value(self.U))
            except RuntimeError:
                pass  # infeasible – ignore

        try:
            sol = self.opti.solve()
            self._prev_sol = sol
            steer = float(sol.value(self.U[0, 0]))
        except RuntimeError as err:
            # solver failed this step – reuse last feasible control
            print("[MPC] solver failed, reusing previous control:", err)
            steer = float(self._prev_sol.value(self.U[0, 0])) if self._prev_sol else 0.0

        return steer, 0.0
