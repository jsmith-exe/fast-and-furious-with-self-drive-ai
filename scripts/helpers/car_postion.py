'''
Prints out the cars relative postion from its starting point, read from the /odom ROS topic
'''

import rospy
from nav_msgs.msg import Odometry
import numpy as np


class CarPositionPrinter:
    def __init__(self, odom_topic='/odom', rate_hz=2):
        self.odom_topic = odom_topic
        self.rate = rospy.Rate(rate_hz)
        self.odom_msg = None
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        print(f"Subscribed to {self.odom_topic} for car position.")

    def odom_callback(self, msg):
        self.odom_msg = msg

    def get_odom_values(self):
        if self.odom_msg is None:
            return None, None, None
        pos = self.odom_msg.pose.pose.position
        x = pos.x
        y = pos.y
        q = self.odom_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        return x, y, theta

    def print_position(self):
        while not rospy.is_shutdown():
            x, y, theta = self.get_odom_values()
            if x is not None:
                print(f"Car position: (x={x:.2f}, y={y:.2f}, theta={np.rad2deg(theta):.1f} deg)")
            else:
                print("No odom message received yet.")
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('print_car_position_from_odom')
    printer = CarPositionPrinter()
    printer.print_position()
