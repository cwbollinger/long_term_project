#! /usr/bin/env python
import rospy

if __name__ == "__main__": # just waits for 5 seconds and then exits
    rospy.init_node('test_node')
    rospy.sleep(5)
