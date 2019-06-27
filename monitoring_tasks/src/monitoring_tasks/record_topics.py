import os
import subprocess

import rospy
import actionlib

import tf

from std_msgs.msg import String

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from long_term_deployment.srv import RequestMap

required_tasks = []


def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed '''

    bag_name = args[0]
    topics = "'{}'".format(' '.join(args[1:]))

    command = ['rosrun', 'monitoring_tasks', 'rosbag_remote_record.py', '-m', 'ros', '-i', topics, '-f', 'default.bag', '-d', '/tmp', '-t', '/rosbagctrl']
    p = subprocess.Popen(command)
    pub = rospy.Publisher('/rosbagctrl/named', String)

    # start recording...
    pub.publish('{}:start'.format(bag_name))

    r = rospy.Rate(10)
    while not stop_event.isSet() and p.poll() is None:
        if stop_event.isSet():
            pub.publish('{}:stop'.format(bag_name))
        r.sleep()

    # stop the bagging process after a delay to make sure things are saved correctly
    rospy.sleep(1)
    p.kill()
