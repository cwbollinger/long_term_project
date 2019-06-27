import rospy

from std_msgs.msg import String

required_tasks = []


def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed '''

    bag_name = args[0]
    print('recording some wifi')

    pub = rospy.Publisher('/rosbagctrl/named', String, queue_size=1)

    rospy.sleep(5)  # make sure the node has time to start
    # start recording...
    pub.publish('{}:start'.format('potato'))
    print('bag file should exist now')

    r = rospy.Rate(10)
    while not stop_event.isSet():
        r.sleep()

    pub.publish('{}:stop'.format('potato'))
    rospy.sleep(0.5)  # small delay to make sure bag file saves ok
