import rospy
from nav_msgs.msg import Odometry
import numpy as np
from evasive_waypoints import compute_global_evasion_waypoint
from scripts.helpers.bottle_live_detection_publish import get_bottle_distance


def get_odom_values(odom_msg):
    pos = odom_msg.pose.pose.position
    x = pos.x
    y = pos.y
    q = odom_msg.pose.pose.orientation
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    theta = np.arctan2(siny_cosp, cosy_cosp)
    return x, y, theta


def main():
    rospy.init_node('evasion_point_from_odom_and_distance')
    odom_msg = None
    lambda_ = 0.5  # Evasion step (sideways) in meters
    rate = rospy.Rate(2)  # 2 Hz
    odom_sub = rospy.Subscriber('/odom', Odometry, lambda msg: None)  # Dummy subscriber to keep topic alive
    odom_topic = '/odom'
    print("Reading /odom and using camera distance. Calculating evasion point...")
    while not rospy.is_shutdown():
        try:
            odom_msg = rospy.wait_for_message(odom_topic, Odometry, timeout=1.0)
        except rospy.ROSException:
            print("No odom message received.")
            rate.sleep()
            continue
        x, y, theta = get_odom_values(odom_msg)
        distance = get_bottle_distance()
        if distance is not None:
            obstacle_pos, evade_pos = compute_global_evasion_waypoint(x, y, theta, distance, lambda_)
            print(f"Car: (x={x:.2f}, y={y:.2f}, theta={np.rad2deg(theta):.1f} deg)")
            print(f"Obstacle: (x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f}), Distance: {distance:.2f} m")
            print(f"Evasion point: (x={evade_pos[0]:.2f}, y={evade_pos[1]:.2f})\n")
        else:
            print("No object detected by camera.")
        rate.sleep()

if __name__ == '__main__':
    main()
