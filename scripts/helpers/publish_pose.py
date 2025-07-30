#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf
import numpy as np
 
def publish_goal_pose(x, y, yaw):
    rospy.init_node('goal_pose_publisher', anonymous=True)
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
 
    # Convert yaw to quaternion
    quat = tf.quaternion_from_euler(0, 0, yaw)  # roll, pitch, yaw
    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]
 
    pub.publish(goal)
    rospy.loginfo("Published goal pose: x=%.2f, y=%.2f, yaw=%.2f rad", x, y, yaw)
 
if __name__ == '__main__':
    try:
        publish_goal_pose(1.0, 1.0, np.deg2rad(-90))  # Example: x=2.0, y=1.0, yaw=90
    except rospy.ROSInterruptException:
        pass