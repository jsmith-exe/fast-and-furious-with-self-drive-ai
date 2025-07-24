import rospy
from std_msgs.msg import Float32
import time

def main():
    rospy.init_node('car_throttle_commander')
    throttle_pub = rospy.Publisher('/throttle', Float32, queue_size=1)
    time.sleep(1)  # Wait for publisher to register
    throttle_value = 0.2
    print(f"Publishing throttle={throttle_value} to /throttle topic...")
    rate = rospy.Rate(10)  # 10 Hz
    try:
        while not rospy.is_shutdown():
            throttle_pub.publish(throttle_value)
            rate.sleep()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping car (throttle=0.0)...")
        throttle_pub.publish(0.0)
        time.sleep(0.1)

if __name__ == '__main__':
    main()
