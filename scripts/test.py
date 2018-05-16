import rospy

if __name__ == "__main__": # just waits for 5 seconds and then exits
    rospy.init_node('test_node')
    r = rospy.Rate(5)
    r.sleep()
