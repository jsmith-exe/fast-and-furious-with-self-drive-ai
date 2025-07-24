import rospy
from geometry_msgs.msg import Twist
import time

def main():
    rospy.init_node('car_cmd_vel_commander')
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    time.sleep(1)  # Wait for publisher to register
    throttle_value = -0.2  # Forward speed (m/s)
    steering_value = 0.0  # No turn (rad/s)
    print(f"Publishing throttle={throttle_value} m/s, steering={steering_value} rad/s to /cmd_vel topic...")
    rate = rospy.Rate(10)  # 10 Hz
    try:
        while not rospy.is_shutdown():
            twist = Twist()
            twist.linear.x = throttle_value
            twist.angular.z = steering_value
            cmd_vel_pub.publish(twist)
            rate.sleep()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping car (cmd_vel=0,0)...")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_vel_pub.publish(twist)
        time.sleep(0.1)

if __name__ == '__main__':
    main()
