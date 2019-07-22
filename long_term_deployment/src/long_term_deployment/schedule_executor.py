#! /usr/bin/env python
import threading

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus

from long_term_deployment.synchronized_actions import SynchronizedSimpleActionClient
from long_term_deployment.msg import Task, TaskGoal, TaskAction
from long_term_deployment.srv import GetSchedule, GetScheduleResponse, AssignSchedule, GetPathLength

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from tf.transformations import quaternion_from_euler, euler_from_quaternion


def heading(q):
    e_angles = euler_from_quaternion([q.w, q.x, q.y, q.z])
    return e_angles[2]


required_tasks = {'navigation_tasks/navigate_on_map': ['graf']}

class ScheduleExecutor(object):

    def __init__(self):
        self.schedule = None
        self.pose_reached = False
        self.curr_task_done = False
        self.curr_task_idx = 0
        self.assign_plan = rospy.Service('~set_schedule', AssignSchedule, self.set_schedule)
        self.provide_plan = rospy.Service('~get_schedule', GetSchedule, self.get_schedule)
        rospy.loginfo('Connecting to local action client...')
        self.active_client = SynchronizedSimpleActionClient(
                '~active',
                TaskAction)
        rospy.loginfo('Connected!')

    def set_schedule(self, msg):
        """ schedule is a time ordered list of tasks to run at locations in the world"""
        self.schedule = msg.schedule
        self.pose_reached = False
        self.curr_task_done = False
        self.curr_task_idx = 0

    def get_schedule(self, msg):
        return GetScheduleResponse(self.schedule)

    def pose_reached_cb(self, term_state, result):
        self.pose_reached = True

    def task_done_cb(self, term_state, result):
        self.curr_task_done = True

    def at_location(self, curr_task):
        return False

    def update(self):

        if self.schedule is None:
            return

        if self.curr_task_idx >= len(self.schedule):
            return # we are done

        curr_task = self.schedule[self.curr_task_idx]
        if self.curr_task_idx != 0:
            prev_task = self.schedule[self.curr_task_idx-1] 
        else:
            prev_task = None

        curr_time = rospy.Time.now()

        if self.curr_task_done:
            self.curr_task_done = False
            self.curr_task_idx += 1

        elif self.pose_reached:
            self.pose_reached = False
            self.active_client.send_goal(TaskGoal(curr_task.task), done_cb=self.task_done_cb)

        elif prev_task is None or prev_task.departure_time <= curr_time <= curr_task.arrival_time:
            if not self.at_location(curr_task) and self.active_client.get_state() != GoalStatus.ACTIVE:
                goal = TaskGoal(Task(
                    workspace_name='',
                    package_name='navigation_tasks',
                    launchfile_name='go_to_pose',
                    args=[
                        str(curr_task.location.position.x),
                        str(curr_task.location.position.y),
                        str(heading(curr_task.location.orientation)),
                    ]
                ))
                self.active_client.send_goal(goal, done_cb=self.pose_reached_cb)

        elif curr_time > curr_task.departure_time:
            self.curr_task_idx += 1


def main(stop_event, args, client_params):

    executor = ScheduleExecutor()
    agent_name = rospy.get_param('~agent_name', 'default')
    expose_local_service = rospy.ServiceProxy('/rosduct/expose_local_service', ROSDuctConnection)

    expose_local_service(conn_name='/robot_client/set_schedule',
                         conn_type='long_term_deployment/AssignSchedule',
                         alias_name='/{}_agent/set_schedule'.format(agent_name))

    expose_local_service(conn_name='/robot_client/get_schedule',
                         conn_type='long_term_deployment/GetSchedule',
                         alias_name='/{}_agent/get_schedule'.format(agent_name))

    expose_local_service(conn_name='/get_path_length',
                         conn_type='long_term_deployment/GetPathLength',
                         alias_name='/{}_agent/get_path_length'.format(agent_name))

    r = rospy.Rate(1)
    while not stop_event.isSet():
        executor.update()
        r.sleep()

    # del executor

    rospy.logerr('exiting')
    return {'status':'success'}


if __name__ == "__main__":
    rospy.init_node('test_node') # can't do this in main or it flips out
    main(threading.Event(), [])
