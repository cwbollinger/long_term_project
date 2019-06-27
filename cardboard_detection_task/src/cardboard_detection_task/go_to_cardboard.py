#!/usr/bin/env python

import os
import subprocess
import threading

import rospy
import actionlib
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from fetch_auto_dock_msgs.msg import DockAction, DockGoal, UndockAction, UndockGoal

import smach
import smach_ros

tag_1_goal = MoveBaseGoal()
tag_1_goal.target_pose.header.frame_id = "map"
tag_1_goal.target_pose.pose.position.x = -14.58
tag_1_goal.target_pose.pose.position.y = 4.10
tag_1_goal.target_pose.pose.orientation.z = -0.705
tag_1_goal.target_pose.pose.orientation.w = 0.709

tag_2_goal = MoveBaseGoal()
tag_2_goal.target_pose.header.frame_id = "map"
tag_2_goal.target_pose.pose.position.x = -0.446
tag_2_goal.target_pose.pose.position.y = 4.338
tag_2_goal.target_pose.pose.orientation.z = -0.735
tag_2_goal.target_pose.pose.orientation.w = 0.6778

dock_goal = MoveBaseGoal()
dock_goal.target_pose.header.frame_id = "map"
dock_goal.target_pose.pose.position.x = -1.0
dock_goal.target_pose.pose.position.y = -0.0
dock_goal.target_pose.pose.orientation.z = 0.0
dock_goal.target_pose.pose.orientation.w = 1.0

goals = [tag_1_goal, tag_2_goal, dock_goal]


def print_status(status):
    g_status = GoalStatus()

    if status == g_status.PREEMPTED:
        print("PREEMPTED")
    elif status == g_status.ABORTED:
        print("ABORTED")
    elif status == g_status.REJECTED:
        print("REJECTED")
    elif status == g_status.SUCCEEDED:
        print("SUCCEEDED")


@smach.cb_interface(input_keys=['counter'], output_keys=['counter', 'classification'], outcomes=['succeeded'])
def cardboard_imaging_cb(ud):
    p = subprocess.Popen([os.path.expanduser('~/long_term_ws/devel/env.sh'), 'roslaunch', 'cardboard_detection_task', 'cardboard_capture_{}.launch'.format(ud.counter+1)])
    while p.poll() is None: # wait until launchfile exits
        pass
    ud.counter += 1

    from cardboard_detection_task import cardboard_query_client
    ud.classification = cardboard_query_client.main(threading.Event(), [])
    #p = subprocess.Popen([os.path.expanduser('~/long_term_ws/devel/env.sh'), 'roslaunch', 'cardboard_detection_task', 'detect_cardboard.launch'])
    #while p.poll() is None: # wait until launchfile exits
    #    pass
    return 'succeeded'


def get_smach_sm():
    sm = smach.StateMachine(outcomes=['succeeded', 'aborted', 'preempted'])
    sm.userdata.counter = 0
    sm.userdata.cardboard_1 = (None, None)
    sm.userdata.cardboard_2 = (None, None)

    with sm:
        smach.StateMachine.add('UNDOCK',
                               smach_ros.SimpleActionState('undock', UndockAction, goal=UndockGoal(rotate_in_place=True)),
                               {'aborted':'GO_TO_CARDBOARD_1'}) # Why does this action always abort when it works every time???

        smach.StateMachine.add('GO_TO_CARDBOARD_1',
                               smach_ros.SimpleActionState('move_base', MoveBaseAction, goal=tag_1_goal),
                               {'succeeded':'TAKE_IMAGE_1',
                                'aborted':'GO_TO_DOCK'})

        smach.StateMachine.add('TAKE_IMAGE_1',
                               smach.CBState(cardboard_imaging_cb),
                               transitions={'succeeded':'GO_TO_CARDBOARD_2'},
                               remapping={'classification':'cardboard_1'})

        smach.StateMachine.add('GO_TO_CARDBOARD_2',
                               smach_ros.SimpleActionState('move_base', MoveBaseAction, goal=tag_2_goal),
                               {'succeeded':'TAKE_IMAGE_2',
                                'aborted':'GO_TO_DOCK'})

        smach.StateMachine.add('TAKE_IMAGE_2',
                               smach.CBState(cardboard_imaging_cb),
                               transitions={'succeeded':'GO_TO_DOCK'},
                               remapping={'classification':'cardboard_2'})

        smach.StateMachine.add('GO_TO_DOCK',
                               smach_ros.SimpleActionState('move_base', MoveBaseAction, goal=dock_goal),
                               {'succeeded':'DOCK'})

        smach.StateMachine.add('DOCK',
                               smach_ros.SimpleActionState('dock', DockAction),
                               {'succeeded':'succeeded'})
    return sm


def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed

    Returns a string representing the return status (json?)
    '''

    feedback_pub = rospy.Publisher('/active_feedback', String, queue_size=10)

    sm = get_smach_sm()

    feedback_pub.publish('Starting state machine in separate thread...')
    smach_thread = threading.Thread(target=sm.execute)
    smach_thread.start()
    
    preempted = False
    r = rospy.Rate(10)
    while smach_thread.isAlive():
        if stop_event.isSet():
            feedback_pub.publish('Task Preempted!')
            preempted = True
            sm.request_preempt()
            break
        r.sleep()
    
    # Block until everything is preempted/completed
    smach_thread.join()
    feedback_pub.publish('State Machine Finished!')
    if not preempted:
        formatted_vals = {
        'waypoint_1': {
            'cardboard': sm.userdata.cardboard_1[0],
            'no_cardboard': sm.userdata.cardboard_1[0]
        },
        'waypoint_2': {
            'cardboard': sm.userdata.cardboard_2[0],
            'no_cardboard': sm.userdata.cardboard_2[0]}
        }
#    else if sm.userdata.cardboard_2[0] is not None:
#        formatted_vals = {
#        'waypoint_1': {'cardboard': sm.userdata.cardboard_1[0], 'no_cardboard': sm.userdata.cardboard_1[0]},
#        'waypoint_2': {'cardboard': sm.userdata.cardboard_2[0], 'no_cardboard': sm.userdata.cardboard_2[0]}
#        }
#    else if sm.userdata.cardboard_1[0] is not None:
#        formatted_vals = {
#        'waypoint_1': {'cardboard': sm.userdata.cardboard_1[0], 'no_cardboard': sm.userdata.cardboard_1[0]},
#        'waypoint_2': {'cardboard': "None", 'no_cardboard': "None"}
#        }
#    else:
#        formatted_vals = {
#        'waypoint_1': {'cardboard': "None", 'no_cardboard': "None"},
#        'waypoint_2': {'cardboard': "None", 'no_cardboard': "None"}
#        }
    return formatted_vals

if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    main(threading.Event(), [])

'''
def movebase_client():
    dock_client = actionlib.SimpleActionClient('dock', DockAction)
    dock_client.wait_for_server()

    undock_client = actionlib.SimpleActionClient('undock', UndockAction)
    undock_client.wait_for_server()

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    u_goal = UndockGoal()
    u_goal.rotate_in_place = True
    undock_client.send_goal(u_goal)
    undock_client.wait_for_result()
    print(undock_client.get_result())
    print_status(undock_client.get_state())

    for i, goal in enumerate(goals):
        client.send_goal(goal)
        wait = client.wait_for_result()
        print("Client Goal:")
        print(client.get_result())
        print_status(client.get_state())
        print("Should have printed")
        if not wait:
            rospy.logger("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            break
        if i == 0 or i == 1: # stupid hacky thing
            # Do Chris's CV thing here
            p = subprocess.Popen([os.path.expanduser('~/long_term_ws/devel/env.sh'), 'roslaunch', 'cardboard_detection_task', 'cardboard_capture_{}.launch'.format(i+1)])
            while p.poll() is None: # wait until launchfile exits
                pass
            p = subprocess.Popen([os.path.expanduser('~/long_term_ws/devel/env.sh'), 'roslaunch', 'cardboard_detection_task', 'detect_cardboard.launch'])
            while p.poll() is None: # wait until launchfile exits
                pass
        elif i == 2:
            dock_client.send_goal(DockGoal())
            dock_client.wait_for_result()
            print(dock_client.get_result())
            print_status(dock_client.get_state())
'''
