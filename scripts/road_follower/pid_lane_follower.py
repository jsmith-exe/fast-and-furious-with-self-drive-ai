#!/usr/bin/env python3

import numpy as np
import time
from pid import PID
#from jetracer.nvidia_racecar import NvidiaRacecar
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

v = 0.75 # Linear Speed

# === PID parameters === ---------------------------------------
KP = 2.0
KI = 0.1 
KD = 0.2 
integral_reset = 0.01
max_steer = np.deg2rad(45)

# --- Globals updated by callback ---
lateral_dev = None
heading_err = 0.0

def line_callback(msg: Float32MultiArray):
    global lateral_dev, heading_err
    if len(msg.data) >= 2:
        lateral_dev = msg.data[0]
        heading_err = msg.data[1]

class PIDFollower():
    def __init__(self):
        self.steering_control = PID(Kp=KP, Ki=KI, Kd=KD, integral_reset=integral_reset, max_value=max_steer)

        rospy.init_node("pid_node", anonymous=True)
        rospy.Subscriber('/line/offset_yaw', Float32MultiArray, line_callback)
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
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
            steer, latency = self.steering_control.update(error=-lateral_dev)

            vel_msg = Twist()

            vel_msg.linear.x = v
            vel_msg.angular.z = steer
            self.pub_vel.publish(vel_msg)

            # DEBUG
            # vel_msg.linear.x = 0.0
            # vel_msg.angular.z = 0.0

            # Periodic Logging
            rospy.loginfo("[PID] Running: Lateral Error=%.3fm, Yaw Error=%.3f°, V=%.2f, ω=%.2f°", lateral_dev, np.rad2deg(heading_err), v, np.rad2deg(steer))


if __name__ == '__main__':
    try:
        system = PIDFollower()
        print("[INFO] PID node ready.")
        print("[INFO] Press Enter to START PID (Ctrl+C to exit).")
        try:
            input()  # <-- waits for you to hit Enter
        except EOFError:
            # If there's no TTY (e.g., roslaunch), just start immediately
            pass

        print("[INFO] Arming… waiting for first line measurement...")
        system.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")
    finally:
        rospy.loginfo("Motors set to zero. Exiting.")
