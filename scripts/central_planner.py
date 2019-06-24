#! /usr/bin/env python

import os
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan

from long_term_deployment.msg import ScheduledTask
from long_term_deployment.srv import ScheduleTask, GetSchedule

import subprocess


class CentralPlanner(object):

    def __init__(self):
        self.op_solver_path = os.path.expanduser('~/catkin_ws/OP_simulator')
        self.schedule_task = rospy.Service('~schedule_task',
                                           ScheduleTask, self.schedule_task)
        self.get_plan = rospy.Service('~get_schedule', GetSchedule, self.get_schedule)
        self.tasks = []

    def schedule_task(self, req):
        rospy.loginfo('Adding new task: {}'.format(req.task))
        self.tasks.append(req.task)
        return True

    def get_schedule(self, req):
        paths = self.get_path_lengths_for_agent(req.agent_name)
        if paths is None:
            return []  # can't compute schedule without edge lengths

        self.write_problem_to_file(paths)
        rospy.loginfo('Calling OP Solver...')
        try:
            exe = os.path.join(self.op_solver_path, 'solve.bash')
            print(exe)
            subprocess.check_output([exe])
        except subprocess.CalledProcessError as e:
            rospy.logerr('Error running OP code: {}'.format(e))
            return []
        except Exception as e:
            rospy.logerr('TERRIBLY WRONG')
            rospy.logerr(e)
        rospy.loginfo('Solution found!')

        schedule = self.read_schedule_from_file()
        return schedule

    def get_path_length(self, path):
        dist = 0
        curr_pos = path.plan.poses[0].pose.position
        for stamped_pose in path.plan.poses:
            dx = curr_pos.x - stamped_pose.pose.position.x
            dy = curr_pos.y - stamped_pose.pose.position.y
            dist += math.sqrt(dx**2 + dy**2)
            curr_pos = stamped_pose.pose.position
        return dist

    def get_path_lengths_for_agent(self, agent_name):
        srv_name = '/{}/move_base/make_plan'.format(agent_name)
        make_plan = rospy.ServiceProxy(srv_name, GetPlan)
        locations = [t.location for t in self.tasks]
        distances = np.ones((len(locations), len(locations)))
        start = PoseStamped()
        start.header.frame_id = 'map'
        start.pose.position.x = 2.0
        end = PoseStamped()
        end.header.frame_id = 'map'
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i == j:
                    continue
                start.pose = loc1
                end.pose = loc2
                try:
                    distances[i, j] = self.get_path_length(make_plan(start, end, 0.2))
                except rospy.ServiceException as e:
                    print('Failure: {}'.format(e))
                    return None

        return distances

    def write_problem_to_file(self, distances):
        distfile = os.path.join(self.op_solver_path, 'distances.npy')
        windowfile = os.path.join(self.op_solver_path, 'windows.npy')
        rospy.loginfo('Writing out problem files: {}, {}'.format(distfile, windowfile))
        windows = np.zeros((len(self.tasks), 3))
        for i, t in enumerate(self.tasks):
            windows[i, 0] = t.start_after.to_time()
            windows[i, 1] = t.finish_before.to_time()
            windows[i, 2] = t.estimated_duration.to_time()
        with open(distfile, 'w') as d, open(windowfile, 'w') as w:
            np.save(d, distances)
            np.save(w, windows)

    def read_schedule_from_file(self):
        sol_file = os.path.join('solution.npy', self.op_solver_path)
        sol_mat = np.load(sol_file)
        solution = []
        for i, t in enumerate(self.tasks):
            s = ScheduledTask()
            s.task = t.task
            s.location = t.location
            s.arrival_time = sol_mat[i, 0]
            s.departure_time = sol_mat[i, 1]
            s.duration = t.estimated_duration
            solution.append(s)

        # reorder by arrival time
        solution.sort(key=lambda t: t.arrival_time)
        return solution

    def remove_past_tasks(self):
        # removes all tasks with departure times after current time from planning list
        for i, task in enumerate(self.tasks):
            if task.finish_before < rospy.Time.now():
                print('Deleting task {}'.format(task))
                del self.tasks[i]


if __name__ == '__main__':
    rospy.init_node('central_planner')
    planner = CentralPlanner()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        planner.remove_past_tasks()
        r.sleep()
