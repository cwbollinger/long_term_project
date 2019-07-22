import rospy

from std_msgs.msg import String

from datetime import datetime

import os


required_tasks = {}

def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed '''

    bag_name = args[0]

    rospy.loginfo('recording wifi')

    bag_pub = rospy.Publisher('/rosbagctrl/named', String, queue_size=1)

    # wait for the node to come up
    r = rospy.Rate(10)
    while bag_pub.get_num_connections() == 0:
        r.sleep()

    ts = rospy.Time.now().to_sec()
    ts_string = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    agent_name = rospy.get_param('~agent_name', 'default')
    bag_name = '{}_{}_wifi'.format(ts_string, agent_name)
    rospy.loginfo(bag_name)

    # start recording...
    bag_pub.publish('{}:start'.format(bag_name))
    rospy.loginfo('bag file should exist now')

    while not stop_event.isSet():
        r.sleep()

    bag_pub.publish('{}:stop'.format(bag_name))
    rospy.sleep(0.5)  # small delay to make sure bag file saves ok
