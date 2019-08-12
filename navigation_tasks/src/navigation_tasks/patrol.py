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

required_tasks = {'navigation_tasks/navigate_on_map': ['graf']}

def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed

    '''

    num_points = len(args) / 3
    points = []
    for i in range(num_points):
        x = float(args[i])
        y = float(args[i+1])
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, float(args[i+2]))
        points.append((x, y, q))

    move_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo('waiting for move_base action server')
    move_client.wait_for_server()
    rospy.loginfo('Server found!')


    status = {'finished': False, 'state': None}  # does not work with plain bool
    def go_next(state, result):
        rospy.loginfo('Action finished, result was:')
        rospy.loginfo('{}'.format(result))

        status['finished'] = True
        status['state'] = state

    r = rospy.Rate(10)
    for p in points:

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose.position.x = p[0]
        goal.target_pose.pose.position.y = p[1]

        goal.target_pose.pose.orientation.x = p[2][0]
        goal.target_pose.pose.orientation.y = p[2][1]
        goal.target_pose.pose.orientation.z = p[2][2]
        goal.target_pose.pose.orientation.w = p[2][3]

        move_client.send_goal(goal, done_cb=go_next)

        while not stop_event.isSet() and not status['finished']:
            r.sleep()
        status['finished'] = False


    if not stop_event.isSet():
        return "{'patrol_status': 'complete'}"
    else:
        move_client.cancel_goal()
        return "{'patrol_status': 'preempted'}"
