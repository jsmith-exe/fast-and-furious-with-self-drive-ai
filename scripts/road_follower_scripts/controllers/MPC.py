import numpy as np
from abc import ABC, abstractmethod

class ControlStrategy(ABC):
    @abstractmethod
    def compute_steering(self, x: np.ndarray) -> float:
        """Given current state vector, return a steering command."""
        pass

class MPC(ControlStrategy):
    """
    Linear MPC with kinematic bicycle dynamics.
    State: [lateral_error, heading_error]
    Dynamics discretized from continuous kinematic bicycle model.
    """
    def __init__(
        self,
        N: int,
        dt: float,
        max_steer: float,
        w_y: float,
        w_psi: float,
        w_delta: float,
        steer_gain: float,
        wheelbase: float,
        velocity: float
    ):
        # horizon, discretization, limits
        self.N = N
        self.DT = dt
        self.MAX_STEER = max_steer
        self.STEER_GAIN = steer_gain
        # vehicle parameters
        self.L = wheelbase
        self.v = velocity

        # Continuous-time kinematic bicycle model
        # x_dot = A_c x + B_c u
        A_c = np.array([[0.0, self.v],
                        [0.0,       0.0]])
        B_c = np.array([[0.0],
                        [self.v/self.L]])

        # Discretize: A = I + A_c*dt, B = B_c*dt
        self.A = np.eye(2) + A_c * self.DT
        self.B = B_c * self.DT

        # Cost weights
        # State cost penalizes lateral (y) and heading (psi)
        self.Q = np.diag([w_y, w_psi])
        # Input cost penalizes steering effort
        self.R = np.array([[w_delta]])

        # Build stacked prediction matrices
        # Phi = [I; A; A^2; ...; A^{N-1}]
        self.Phi = np.vstack([np.linalg.matrix_power(self.A, i) for i in range(N)])
        # Gamma is block lower-triangular
        n_state = 2
        self.Gamma = np.zeros((n_state*N, N))
        for k in range(1, N):
            for j in range(k):
                self.Gamma[k*n_state:(k+1)*n_state, j] = (
                    np.linalg.matrix_power(self.A, k-1-j) @ self.B
                ).flatten()

        # Build cost matrices
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)
        # Hessian H = 2*(Gamma^T Qbar Gamma + Rbar)
        self.H = 2*(self.Gamma.T @ Qbar @ self.Gamma + Rbar)
        # Precompute inverse or factorization
        self.H_inv = np.linalg.inv(self.H)

        # Precompute gradient constant: M = Gamma^T Qbar Phi
        self.M = self.Gamma.T @ Qbar @ self.Phi

    def compute_steering(self, x: np.ndarray) -> float:
        """
        x: state vector [y_error, psi_error]
        Returns first steering move in receding-horizon.
        """
        # Gradient term g = 2 * M * x0
        g = 2 * (self.M @ x)

        # Solve for delta sequence
        delta_seq = - self.H_inv @ g
        delta0 = float(delta_seq[0])

        # Clip and scale
        delta0_clipped = np.clip(delta0, -self.MAX_STEER, self.MAX_STEER)
        return -delta0_clipped * self.STEER_GAIN
