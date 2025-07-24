import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np


def odom_callback(msg):
    pos = msg.pose.pose.position
    print(f"Car position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")

def imu_callback(msg):
    q = msg.orientation
    # Convert quaternion to yaw (theta)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    theta = np.arctan2(siny_cosp, cosy_cosp)
    print(f"IMU yaw (theta): {np.rad2deg(theta):.2f} deg")

def main():
    rospy.init_node('car_position_printer')
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/imu', Imu, imu_callback)
    print("Subscribed to /odom and /imu. Printing car position and IMU yaw...")
    rospy.spin()

if __name__ == '__main__':
    main()
