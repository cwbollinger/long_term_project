#! /usr/bin/env python

import os
import math
import rospy

from long_term_deployment.srv import GetPathLength, GetPathLengthResponse
from nav_msgs.srv import GetPlan

import subprocess
import datetime


class PathLengthServer(object):

    def __init__(self):
        self.path_srv = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        self.length_server = rospy.Service('/get_path_length', GetPathLength, self.length_service_cb)

    def length_service_cb(self, msg):
        path = self.path_srv(msg.start, msg.goal, msg.tolerance)
        length = self.get_path_length(path)
        rospy.logwarn('Path Length: {}'.format(length))
        return GetPathLengthResponse(length)

    def get_path_length(self, path):
        dist = 0
        curr_pos = path.plan.poses[0].pose.position
        for stamped_pose in path.plan.poses:
            dx = curr_pos.x - stamped_pose.pose.position.x
            dy = curr_pos.y - stamped_pose.pose.position.y
            dist += math.sqrt(dx**2 + dy**2)
            curr_pos = stamped_pose.pose.position
        return dist


if __name__ == '__main__':
    rospy.init_node('path_length_server')
    srv = PathLengthServer()
    rospy.spin()
