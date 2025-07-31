#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import transformations as tf
import numpy as np
from scripts.collision_avoidance.evasion_point_live import EvasionPointStreamer
 
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
 
    # Convert yaw to quaternion
    quat = tf.quaternion_from_euler(0, 0, yaw)  # roll, pitch, yaw
    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]
 
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

            publish_goal_pose(0, 0, np.deg2rad(170))  # Example: x=2.0, y=1.0, yaw=90

            '''if evade_pos is not None and distance > 0.1:

                #print(f"Evasion point: (x={evade_pos[0]:.2f}, y={evade_pos[1]:.2f})")
                #print(f"Return point:  (x={return_pos[0]:.2f}, y={return_pos[1]:.2f})")
                # Publish the evade position as a goal pose
                #print(evade_pos)
                publish_goal_pose(evade_pos[0], evade_pos[1], theta)  # Example: x=2.0, y=1.0, yaw=90
                #publish_goal_pose(return_pos[0], return_pos[1], theta)  # Example: x=2.0, y=1.0, yaw=90
                #break'''

    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")