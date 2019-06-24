import os
import subprocess

import rospy
import actionlib
from actionlib import CommState

import tf

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from long_term_deployment.srv import RequestMap

required_tasks = ['navigation_tasks/navigate_on_map']

def main(stop_event, args):
    ''' Takes a threading.Event to know if preemption is needed

    '''

    move_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    print('waiting for move_base action server')
    move_client.wait_for_server()
    print('Server found!')

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.pose.position.x = float(args[0])
    goal.target_pose.pose.position.y = float(args[1])

    q = tf.transformations.quaternion_from_euler(0.0, 0.0, float(args[2]))
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]

    status = {'finished': False, 'state': None}  # does not work with plain bool
    def finished(state, result):
        status['finished'] = True
        status['state'] = state

    move_client.send_goal(goal, done_cb=finished)

    r = rospy.Rate(10)
    while not stop_event.isSet() and not status['finished']:
        # print(stop_event.isSet(), status['finished'])
        r.sleep()

    if not stop_event.isSet():
        return "{'new_pose': "+str(args)+"}"
