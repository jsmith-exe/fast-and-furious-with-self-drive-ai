from time import sleep

from scripts.helpers.jetracer_class import JetracerInitializer

class CollisionAvoidance:
    def __init__(self, jetracer: JetracerInitializer) -> None:
        self.jetracer = jetracer
        self.car = jetracer.car

        # Obstacle avoidance parameters
        self.turning_away_duration = jetracer.turning_away_duration
        self.turning_back_duration = jetracer.turning_back_duration
        self.steering_away_value = jetracer.steering_away_value
        self.steering_back_value = jetracer.steering_back_value

    def right_turn(self) -> None:
        self.jetracer.car.throttle = 0.5
        turning_away_duration = 0.36 / self.jetracer.car.throttle
        turning_back_duration = 0.36 / self.jetracer.car.throttle 
        self.car.steering = self.steering_away_value
        sleep(turning_away_duration)
        self.car.steering = self.steering_back_value
        sleep(turning_back_duration)

    def left_turn(self) -> None:
        self.jetracer.car.throttle = 0.5
        turning_back_duration = 0.36 / self.jetracer.car.throttle
        turning_away_duration = 0.36 / self.jetracer.car.throttle
        self.car.steering = -self.steering_away_value
        sleep(turning_away_duration)
        self.car.steering = -self.steering_back_value
        sleep(turning_back_duration)
