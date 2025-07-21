import os, sys, turtle, numpy as np
from typing import Tuple, Union
from controllers.PID import PID
from controllers.MPC import MPC

class LineFollowerSim:
    def __init__(self,
                 controller: Union[PID, MPC],
                 course_length: int = 1400,
                 car_velocity: float = 1.0,
                 car_starting_offset: float = 0,
                 course_amplitude: float = 0.5,
                 course_freq: float = 0.7,
                 dt: float = 0.05,
                 max_steer_degrees: float = 40,
                 line_detection: str = "cte",
                 camera_max_centimeters: float = 15,
                 camera_scale: float = 4.2):
        # --- Make sure Tcl/Tk is found (needed for turtle on some installs) ---
        base_prefix = getattr(sys, 'base_prefix', sys.prefix)
        os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
        os.environ['TK_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tk8.6')

        self.ctrl = controller
        self.course_amp = course_amplitude
        self.course_freq = course_freq
        self.course_length = course_length
        self.start_position_x = course_length / 2
        self.start_position_meters = course_length / 200  # starting x-position (m)
        self.v = car_velocity
        self.car_starting_offset = car_starting_offset * 100
        self.dt = dt
        self.max_steer = np.deg2rad(max_steer_degrees)
        self.line_detection = line_detection
        self.camera_max_centimeters = camera_max_centimeters
        self.cte = car_starting_offset
        self.camera = 0
        self.car_x_position = -course_length / 200
        self.camera_scale = camera_scale

        # Instantiation of the path module
        self.screen = turtle.Screen()
        self.screen.setup(course_length, 600)
        self.screen.title('Line follower')
        self.path = turtle.Turtle(visible=False)

        self.draw_the_course()

        # Instantiation of the car module
        self.car = turtle.Turtle()
        self.car.shape('arrow')
        self.car.color('red')

        self.place_the_car_on_the_course()

    def course_function(self, x: float):
        "Desired y position for a given x (simple sine wave)."
        return self.course_amp * np.sin(self.course_freq * x)

    def convert_meters_to_centimeters(self, meters: float) -> float:
        return meters * 100

    # === Turtle visualisation (unchanged) ========================================
    def draw_the_course(self) -> None:
        # Draw the reference path once
        self.path.penup()
        self.path.goto(-self.start_position_x, self.course_function(-self.start_position_meters) * 100)
        self.path.pendown()
        for px in range(int(-self.start_position_x), int(self.start_position_x + 1)):
            xm = px / 100.0
            self.path.goto(px, self.course_function(xm) * 100)

    def place_the_car_on_the_course(self) -> None:
        self.car.penup()
        self.car.goto(-self.start_position_x, self.course_function(-self.start_position_meters) * 100 + self.car_starting_offset)
        self.car.setheading(0)
        self.car.pendown()

    def get_steering_value(self, ctrl: Union[PID, MPC], cte: float) -> Tuple[float, float]:
        """
            Compute a steering command from either a PID or MPC controller.
        """
        if isinstance(ctrl, PID):
            steer = ctrl.compute_steering(-cte)  # scalar in, scalar out
            latency = 0.0
        else:  # MPC
            steer, latency = ctrl.compute_steering([cte])  # list/array state
        return steer, latency

    def compute_cte(self) -> None:
        y_real = self.car.ycor() / 100 # turtle y (m)
        self.cte = y_real - self.course_function(self.car_x_position)

    def compute_camera(self) -> None:
        cte_centimeters = self.convert_meters_to_centimeters(meters=self.cte)
        cte_clipped = np.clip(a=cte_centimeters, a_min=-self.camera_max_centimeters, a_max=self.camera_max_centimeters)
        self.camera = cte_clipped / self.camera_max_centimeters

    def get_camera_offset(self) -> float:
        return self.camera / self.camera_scale

    # === Main loop ===============================================================
    def run(self, steps: int = 400) -> None:
        for step in range(steps):
            self.compute_cte()
            self.compute_camera()

            track_offset = self.cte

            if self.line_detection == "camera":
                track_offset = self.get_camera_offset()

            if isinstance(self.ctrl, PID):
                track_offset = -track_offset

            # --- get steering command from the chosen controller ------------------
            steer, latency = self.ctrl.compute_steering(error=track_offset)
            steer = np.clip(steer, -self.max_steer, self.max_steer)

            # Log & visualise
            print(f"step {step:3d} | CTE={self.cte:+.3f} m | camera={self.camera:+.3f} | camera offset value={self.get_camera_offset():+3f} | steer={np.rad2deg(steer):+.1f}°"
                  f" | solver latency={latency:.1f} ms")

            self.car.setheading(np.rad2deg(steer))
            self.car.forward(self.v * 100 * self.dt)  # 100 px ≈ 1 m
            self.car_x_position += self.v * self.dt

        turtle.done()