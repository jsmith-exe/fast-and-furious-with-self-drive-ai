#!/usr/bin/env python3

import numpy as np
import time
from scripts.road_follower.pid import PID
#from jetracer.nvidia_racecar import NvidiaRacecar
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

speed = 0.3 # Linear Speed

# === PID parameters === ---------------------------------------
KP = 0.85
KI = 0.1 
KD = 0.2 
integral_reset = 0.01
max_steer = np.deg2rad(45)

def line_callback(msg: Float32MultiArray):
    global lateral_dev, heading_err
    if len(msg.data) >= 2:
        lateral_dev = msg.data[0]
        heading_err = msg.data[1]

class PIDFollower():
    def __init__(self):
        # # Car setup
        # self.car = NvidiaRacecar()
        # self.car.steering_gain = -1.0    # no additional gain here
        # self.car.throttle     = 0
        # self.throttle_gain = throttle_gain
        self.steering_control = PID(Kp=KP, Ki=KI, Kd=KD, integral_reset=integral_reset, max_value=max_steer)

        rospy.init_node("pid_node", anonymous=True)
        rospy.Subscriber('/line/offset_yaw', Float32MultiArray, line_callback)
        pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.r = rospy.Rate(20)
        print("[INFO] Init Node...")
        print("[INFO] Wait 2s ...")    
        st = time.time()
        while time.time() - st < 2:
            self.r.sleep()
            continue   
        print("[INFO] Start PID !!!")

    

    def run(self) -> None:
        while not rospy.is_shutdown():
            if lateral_dev is None:
                rospy.logwarn_throttle(5, "[PID] Waiting for Line Measurements...")
                self.r.sleep()
                continue

            # Compute steering via PID controller
            steer, latency = self.ctrl.compute_steering(error=lateral_dev)

            vel_msg = Twist()

            vel_msg.linear.x = speed
            vel_msg.angular.z = steer
            self.pub_vel.publish(vel_msg)

            # DEBUG
            # vel_msg.linear.x = 0.0
            # vel_msg.angular.z = 0.0

            # steer_clipped = np.clip(steer, -1, 1)
            # self.car.steering = steer_clipped

            # self.car.throttle = self.jetracer.throttle_gain

            # Periodic Logging
            rospy.loginfo("[PID] Running: Lateral Error=%.3fm, Yaw Error=%.3f°, V=%.2f, ω=%.2f°", lateral_dev, np.rad2deg(heading_err), speed, np.rad2deg(steer))


if __name__ == '__main__':
    try:
        system = PIDFollower()
        system.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")
    finally:
        rospy.loginfo("Motors set to zero. Exiting.")
