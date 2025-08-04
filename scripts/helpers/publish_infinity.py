#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from math import sqrt, atan2, sin, cos
import numpy as np
import math

def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Return a geometry_msgs/Quaternion representing a rotation of `yaw` around the Z axis.
    """
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)
    return Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))

# Convert quaternion to euler
def quaternion2Yaw(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    yaw = math.atan2(2.0*(w*z + x*y), 1.0-2.0*(y**2 + z**2))
    return yaw

def infinity_waypoints(a=1.0, n=100):
    """
    Compute a lemniscate ("∞") trajectory.
    Returns arrays xs, ys, yaws of length n.
    """
    ts = np.linspace(0, 2 * np.pi, n)
    xs = a * np.sin(ts) / (1 + np.cos(ts)**2)
    ys = a * np.sin(ts) * np.cos(ts) / (1 + np.cos(ts)**2)
    # heading is tangent: arctan2(dy/dt, dx/dt)
    dx = np.gradient(xs, ts)
    dy = np.gradient(ys, ts)
    yaws = np.arctan2(dy, dx)
    yaws = np.unwrap(yaws)  # unwrap to avoid discontinuities
    return list(zip(xs, ys, yaws))

# parameters
DIST_THRESH = 0.2    
ANGLE_THRESH = np.deg2rad(5)
WAYPOINTS = infinity_waypoints(a=2, n=100)

# globals
current_pose = None
wp_index = 0

def odom_cb(msg):
    global current_pose
    px = msg.pose.pose.position.x
    py = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    # quaternion → yaw
    yaw = quaternion2Yaw(q)
    current_pose = (px, py, yaw)

def publish_goal(x, y, yaw):
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0
    # quaternion about z
    qz = sin(yaw/2.0)
    qw = cos(yaw/2.0)
    goal.pose.orientation = yaw_to_quaternion(yaw)
    pub.publish(goal)
    rospy.loginfo(f"→ Published waypoint {wp_index}: ({x:.2f},{y:.2f}, yaw={np.rad2deg(yaw):.1f}°)")

if __name__ == "__main__":
    rospy.init_node("infinity_pose_publisher")
    rospy.Subscriber("/odom", Odometry, odom_cb)
    pub = rospy.Publisher("/nmpc/goal_pose", PoseStamped, queue_size=1)
    rospy.sleep(1.0)

    rate = rospy.Rate(10)  # 10 Hz loop
    # publish the 1st waypoint
    xg, yg, ygaw = WAYPOINTS[wp_index]
    publish_goal(xg, yg, ygaw)

    while not rospy.is_shutdown() and wp_index < len(WAYPOINTS):
        if current_pose is None:
            rate.sleep()
            continue

        xr, yr, ryaw = current_pose
        xg, yg, ygaw = WAYPOINTS[wp_index]
        error = np.linalg.norm([xr-xg, yr-yg])
        delta = ryaw - ygaw
        error_angle = math.atan2(np.sin(delta), np.cos(delta))
        dx, dy = xr - xg, yr - yg
        dist = sqrt(dx*dx + dy*dy)
        ang_err = atan2(sin(ryaw-ygaw), cos(ryaw-ygaw))

        if error < DIST_THRESH and abs(error_angle) < ANGLE_THRESH:
            # move to next waypoint
            wp_index += 1
            if wp_index < len(WAYPOINTS):
                xg, yg, ygaw = WAYPOINTS[wp_index]
                publish_goal(xg, yg, ygaw)
        rate.sleep()
