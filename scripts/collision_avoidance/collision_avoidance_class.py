from time import sleep

from scripts.helpers.jetracer_class import JetracerInitializer

class CollisionAvoidance:
    def __init__(self, jetracer: JetracerInitializer) -> None:
        self.jetracer = jetracer
        self.car = jetracer.car

        # Camera and model setup
        self.camera = jetracer.camera
        self.model = jetracer.model

        # Controller strategy
        self.ctrl = jetracer.ctrl

        # Obstacle avoidance parameters
        self.turning_away_duration = jetracer.turning_away_duration
        self.turning_back_duration = jetracer.turning_back_duration
        self.steering_away_value = jetracer.steering_away_value
        self.steering_back_value = jetracer.steering_back_value

    def right_turn(self) -> None:
        self.car.steering = self.steering_away_value
        sleep(self.turning_away_duration)
        self.car.steering = self.steering_back_value

    def left_turn(self) -> None:
        self.car.steering = -self.steering_away_value
        sleep(self.turning_away_duration)
        self.car.steering = -self.steering_back_value
