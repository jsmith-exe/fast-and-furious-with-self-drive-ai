#!/usr/bin/env python3
'''
Listens to the ROS topic /line/offset_yaw , gets the values of Lateral deviation and Yaw.
The NMPC takes these values, proccess and output.
The output is packaged in a twist message and posted to the ROS topic /cmd_vel.
'''
import numpy as np
import math
import time
from nmpc import NMPC

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# === MPC parameters === ---------------------------------------
N = 20
DT = 0.1
lateral_dev_weight = 200
yaw_weight = 3
min_v = 0.4    # min speed the nmpc can output
max_v = 1.2     # max speed the nmpc can output
max_turning_angle = np.deg2rad(45)  # max turning angle of jetracer (degrees)


nmpc_active = True
def nmpc_toggle_callback(msg):
    global nmpc_active
    nmpc_active = msg.data


def line_callback(msg: Float32MultiArray):
    global lateral_dev, heading_err
    if len(msg.data) >= 2:
        lateral_dev = msg.data[0]
        heading_err = msg.data[1]

# Convert quaternion to euler
def quaternion2Yaw(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    yaw = math.atan2(2.0*(w*z + x*y), 1.0-2.0*(y**2 + z**2))
    return yaw

def desired_trajectory(fb, N: int, pose_ref):
    goal = pose_ref
    ref_ = np.zeros((N+1, 2))
    ct_ = np.zeros((N, 2))
    delta_yaw = math.atan2(math.sin(goal[1] - fb[1]), math.cos(goal[1] - fb[1]))
    for i in range(N + 1):
        alpha = i / N
        ref_[i, :1] = fb[:1] * (1 - alpha) + goal[:1] * alpha
        interp_yaw = fb[1] + alpha * delta_yaw
        ref_[i, 1] = math.atan2(math.sin(interp_yaw), math.cos(interp_yaw))
    return ref_, ct_

def nmpc_node():
    rospy.init_node("nmpc_node", anonymous=True)
    rospy.Subscriber('/line/offset_yaw', Float32MultiArray, line_callback)
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    r = rospy.Rate(20)
    print("[INFO] Init Node...")
    print("[INFO] Wait 2s ...")    
    st = time.time()
    while time.time() - st < 2:
        r.sleep()
        continue
    print("[INFO] NMPC node ready.")
    print("[INFO] Press Enter to START NMPC (Ctrl+C to exit).")
    try:
        input()  # <-- waits for you to hit Enter
    except EOFError:
        # If there's no TTY (e.g., roslaunch), just start immediately
        pass

    print("[INFO] Arming… waiting for first line measurement...")  

    # Create the current robot position
    pos_fb = np.array([lateral_dev, heading_err]) 
    W_q = np.diag([lateral_dev_weight, yaw_weight])  
    W_r = np.diag([1, 1])  
    W_v = 10**2*np.diag([1, 1]) 
    W_dv = np.diag([100, 100])  
    nmpc = NMPC(pos_fb, DT, N, W_q, W_r, W_v, W_dv, min_v, max_v, max_turning_angle)

    # compute reference
    target = np.array([0, 0])

    while not rospy.is_shutdown():
        if lateral_dev is None:
            rospy.logwarn_throttle(5, "[NMPC] Waiting for Line Measurements...")
            r.sleep()
            continue

        # Create the current robot position
        pos_fb = np.array([lateral_dev, heading_err])

        next_traj, next_cons = desired_trajectory(pos_fb, N, target)
        st = time.time()
        vel = nmpc.solve(next_traj, next_cons)
        #print("Processing time: {:.2f}s".format(time.time()-st))

        # rospy.loginfo(f"[NMPC] x_fb={x_fb:.2f}, y_fb={y_fb:.2f}, yaw_fb={np.rad2deg(yaw_fb):.1f}°")
        # rospy.loginfo(f"[NMPC] x_ref={x_ref:.2f}, y_ref={y_ref:.2f}, yaw_ref={np.rad2deg(yaw_ref):.1f}°")
        # rospy.loginfo(f"[NMPC] error={error:.3f} m, error_angle={np.rad2deg(error_angle):.1f}°")
        vel_msg = Twist()
        V_MIN_CAP = 0.1
        
        if abs(vel[0]) < V_MIN_CAP:
            vel[0] = V_MIN_CAP if vel[0] > 0 else -V_MIN_CAP
        
        # angle_deg_arr = np.rad2deg(next_traj[2])   # still an array
        # angle_deg = float(angle_deg_arr[0])          # now a plain Python float
        # rospy.loginfo("[NMPC] Running: angle=%.3fm", angle_deg)

        rospy.loginfo("[NMPC] Running: Lateral Error=%.3fm, Yaw Error=%.3f°, V=%.2f, ω=%.2f°", lateral_dev, np.rad2deg(heading_err), vel[0], np.rad2deg(vel[1]))

        vel_msg.linear.x = vel[0]
        vel_msg.angular.z = vel[1]
        # DEBUG
        # vel_msg.linear.x = 0.0
        # vel_msg.angular.z = 0.0
            
        pub_vel.publish(vel_msg)
 
        r.sleep()
if __name__ == '__main__':    
    try:
        nmpc_node()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")
    finally:
        rospy.loginfo("Motors set to zero. Exiting.")
        
        
