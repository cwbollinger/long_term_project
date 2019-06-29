#! /usr/bin/env python

import os
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan

from long_term_deployment.msg import ScheduledTask
from long_term_deployment.srv import ScheduleTask, GetSchedule, GetScheduleResponse

import subprocess
import datetime


class CentralPlanner(object):

    def __init__(self):
        self.op_solver_path = os.path.expanduser('~/long_term_ws/OP_simulator')
        self.schedule_task = rospy.Service('~schedule_task',
                                           ScheduleTask, self.schedule_task)
        self.get_plan = rospy.Service('~get_schedule', GetSchedule, self.get_schedule)
        self.tasks = []

    def schedule_task(self, req):
        rospy.loginfo('Adding new task at:\n{}'.format(req.task.location.position))
        self.tasks.append(req.task)
        return True

    def get_schedule(self, req):
        self.distances = self.get_path_lengths_for_agent(req.agent_name)
        if self.distances is None:
            return []  # can't compute schedule without edge lengths

        file_paths = self.write_problem_to_file()
        file_paths = self.write_problem_to_file()
        rospy.loginfo('Calling OP Solver...')
        try:
            script = os.path.join(self.op_solver_path, 'time_window_service_times_solver.py')
            cmd = ['python3', script] + file_paths
            print(cmd)
            subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            rospy.logerr('Error running OP code: {}'.format(e))
            return []
        except Exception as e:
            rospy.logerr('TERRIBLY WRONG')
            rospy.logerr(e)
            return []
        rospy.loginfo('Solution found!')

        schedule = self.read_schedule_from_file()
        # print(schedule)
        return GetScheduleResponse(schedule)

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
                    start_pose = (start.pose.position.x, start.pose.position.y)
                    end_pose = (end.pose.position.x, end.pose.position.y)

                    rospy.loginfo('Getting distance from {} to {} '.format(start_pose, end_pose))
                    path = make_plan(start, end, 0.2)
                    distances[i, j] = self.get_path_length(path)
                    rospy.loginfo('Distance: {}'.format(distances[i, j]))
                    
                except rospy.ServiceException as e:
                    rospy.logerr('Failure: {}'.format(e))
                    return None

        return distances

    def write_problem_to_file(self):
        distfile = os.path.join(self.op_solver_path, 'distances.npy')
        windowfile = os.path.join(self.op_solver_path, 'windows.npy')
        rospy.loginfo('Writing out problem files: {}, {}'.format(distfile, windowfile))
        windows = np.zeros((len(self.tasks), 3))
        for i, t in enumerate(self.tasks):
            windows[i, 0] = t.start_after.to_sec()
            windows[i, 1] = t.finish_before.to_sec()
            windows[i, 2] = t.estimated_duration.to_sec()
        with open(distfile, 'w') as d, open(windowfile, 'w') as w:
            # TODO: Uncomment this once time issues are resolved
            np.save(d, self.distances)
            np.save(w, windows)
        return [distfile, windowfile]

    def read_schedule_from_file(self):
        sol_file = os.path.join(self.op_solver_path, 'solution.npy')
        sol_mat = np.load(sol_file)
        solution = []
        for i, t in enumerate(self.tasks):
            s = ScheduledTask()
            s.task = t.task
            s.location = t.location
            # TODO: Figure out how to compute duration if possible.
            s.duration = self.tasks[i].estimated_duration
            s.arrival_time = rospy.Time.from_sec(sol_mat[i, 0])
            s.departure_time = rospy.Time.from_sec(sol_mat[i, 1])
            solution.append(s)

        # for idx in range(len(sol_mat)/2):
        #     n = int(sol_mat[2*idx, 0])
        #     # print(n)
        #     # print(type(rospy.Time.from_sec(sol_mat[2*idx, 1])))
        #     # print(rospy.Time.from_sec(sol_mat[2*idx, 1]))
        #     # print(type(rospy.Time.from_sec(sol_mat[2*idx+1, 1])))
        #     # print(rospy.Time.from_sec(sol_mat[2*idx+1, 1]))
        #     solution[n].arrival_time = rospy.Time.from_sec(sol_mat[2*idx, 1])
        #     solution[n].departure_time = rospy.Time.from_sec(sol_mat[2*idx+1, 1])

        #     dur = rospy.Duration.from_sec(sol_mat[2*idx+1, 1] - sol_mat[2*idx, 1])
        #     solution[n].duration = dur

        #     def readable_time(t):
        #         return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
        #     print(readable_time(solution[n].arrival_time.to_sec()))
        #     print(readable_time(solution[n].departure_time.to_sec()))
        #     print(solution[n].duration.to_sec())

        # print(sol_mat)

        # reorder by arrival time
        solution.sort(key=lambda t: t.arrival_time)
        return solution

    def remove_past_tasks(self):
        # removes all tasks with departure times after current time from planning list
        for i, task in enumerate(self.tasks):
            if task.finish_before < rospy.Time.now():
                rospy.loginfo('Deleting expired task {}'.format(task))
                del self.tasks[i]


if __name__ == '__main__':
    rospy.init_node('central_planner')
    planner = CentralPlanner()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        planner.remove_past_tasks()
        r.sleep()
