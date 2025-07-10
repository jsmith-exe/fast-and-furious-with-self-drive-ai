#!/usr/bin/env python2
import rospy
from nav_msgs.msg import Odometry
import tf

def odom_callback(msg):
    # Position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z

    # Orientation (quaternion: roll, pitch, yaw)
    q = msg.pose.pose.orientation
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w]
    )

    # Linear velocity
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z

    # Angular velocity
    wx = msg.twist.twist.angular.x
    wy = msg.twist.twist.angular.y
    wz = msg.twist.twist.angular.z

    rospy.loginfo("Position:  x={:.3f}, y={:.3f}, z={:.3f}".format(x, y, z))
    rospy.loginfo("Orientation (rpy): roll={:.3f}, pitch={:.3f}, yaw={:.3f}".format(roll, pitch, yaw))
    rospy.loginfo("Linear Vel:  vx={:.3f}, vy={:.3f}, vz={:.3f}".format(vx, vy, vz))
    rospy.loginfo("Angular Vel: wx={:.3f}, wy={:.3f}, wz={:.3f}".format(wx, wy, wz))
    rospy.loginfo("---------------------------------------------")

def listener():
    rospy.init_node('odom_reader_py2', anonymous=True)
    rospy.Subscriber('/odom', Odometry, odom_callback, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
