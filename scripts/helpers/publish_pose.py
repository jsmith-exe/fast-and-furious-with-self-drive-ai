#!/usr/bin/env python3

'''
Publish the pose and orientation of the JetRacer to the ROS topic /nmpc/goal_pose
'''

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
import numpy as np
from scripts.collision_avoidance.evasion_point_live import EvasionPointStreamer

# Pose position and orientation
x = 0               # (m)
y = 0               # (m)
orientation_degrees = 180   # (degrees)

def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Return a geometry_msgs/Quaternion representing a rotation of `yaw` around the Z axis.
    """
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)
    return Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
 
def publish_goal_pose(x, y, yaw):
    #rospy.init_node('goal_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/nmpc/goal_pose', PoseStamped, queue_size=10)
 
    # Wait for the publisher to connect
    rospy.sleep(1)
 
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"  # or "odom" depending on your setup
 
    # Set position
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0

    goal.pose.orientation = yaw_to_quaternion(yaw)
 
    pub.publish(goal)
    rospy.loginfo("Published goal pose: x=%.2f, y=%.2f, yaw=%.2f deg", x, y, np.rad2deg(yaw))

if __name__ == '__main__':
    rospy.init_node('evasion_point_data')
    try:
        evasion = EvasionPointStreamer()
        while not rospy.is_shutdown():
            x, y, theta = evasion.get_car_position()
            distance, bbox = evasion.get_object_position()
            obstacle_pos, evade_pos, return_pos = evasion.process_evasion_point(x, y, theta, distance)

            publish_goal_pose(x, y, np.deg2rad(orientation_degrees)) 

    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")