import numpy as np
from abc import ABC, abstractmethod

class ControlStrategy(ABC):
    @abstractmethod
    def compute_steering(self, x_norm: float) -> float:
        """Given normalized lateral error, return a steering command."""
        pass

class MPC(ControlStrategy):
    """
    Analytical N-step Model Predictive Controller for lateral guidance
    on a simple integrator model, solved in closed form.
    """
    def __init__(
        self,
        N: int,
        dt: float,
        max_steer: float,
        w_y: float,
        w_delta: float,
        steer_gain : float
    ):
        # horizon length
        self.N = N
        # timestep and steering limits
        self.DT = dt
        self.MAX_STEER = max_steer
        # weights for cross-track error and steering effort
        self.W_Y = w_y
        self.W_DELTA = w_delta
        self.STEER_GAIN = steer_gain

        # Precompute sums for analytic solution
        # sum_{k=0 to N-1} k = N*(N-1)/2
        self.sum_k = N * (N - 1) / 2.0
        # sum_{k=0 to N-1} k^2 = (N-1)*N*(2N-1)/6
        self.sum_k2 = (N - 1) * N * (2 * N - 1) / 6.0

    def compute_steering(self, x_norm: float) -> float:
        """
        Closed-form N-step MPC:
        minimizes J = sum_{k=0 to N-1} [W_Y*(y_k)^2 + W_DELTA*(delta)^2]
        subject to y_k = x_norm + DT * sum_{i=0 to k-1} delta
        => optimal constant delta = -(W_Y*DT*x_norm*sum_k) /
                                   (W_Y*DT^2*sum_k2 + W_DELTA*N)
        """
        # numerator and denominator for constant control over horizon
        num = - self.W_Y * self.DT * x_norm * self.sum_k
        den = self.W_Y * (self.DT ** 2) * self.sum_k2 + self.W_DELTA * self.N
        delta_opt = num / den
        # enforce steering limits
        steer = float(np.clip(delta_opt, -self.MAX_STEER, self.MAX_STEER))
        return -steer * self.STEER_GAIN

