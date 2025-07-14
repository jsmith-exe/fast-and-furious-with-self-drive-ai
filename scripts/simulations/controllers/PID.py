import time
import numpy as np
from abc import ABC, abstractmethod

class ControlStrategy(ABC):
    @abstractmethod
    def compute_steering(self, x_norm: float) -> float:
        """Given normalized lateral error, return a steering command."""
        pass

class PID(ControlStrategy):
    def __init__(self, Kp: float, Ki: float, Kd: float, integral_reset: float = None, delay: float = 0.15, max_value: float = np.inf):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0
        self.prev_error = 0
        self.last_time = time.time()

        self.delay = delay
        self.integral_reset = integral_reset
        self.prev_value = 0

        self.max_value = max_value

    def compute_steering(self, error: float):
        dt = (time.time() - self.last_time)
        if dt < self.delay:
            return self.prev_value

        if self.integral_reset is not None:
            if abs(error) <= self.integral_reset:
                self.integral = 0

        self.integral += error * dt
        if dt == 0:
            derivative = 0
        else:
            derivative =  (error - self.prev_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(-self.max_value, output, self.max_value)
        #print(f"prev: {self.prev_error}, error: {error}, dt: {dt},derivative: {derivative}")
        #print(f"Output: {output}, P: {self.Kp*error}, I: {self.Ki*self.integral}, D: {self.Kd*derivative}, dt: {dt}, error: {error}, prev error: {self.prev_error}")

        self.prev_error = error
        self.last_time = time.time()

        #time.sleep(self.delay)
        self.prev_value = output
        return output