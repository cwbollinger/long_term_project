#! /usr/bin/env python

import rospy
import actionlib
from frontier_exploration.msg import ExploreTaskAction, ExploreTaskGoal

if __name__ == "__main__":
    rospy.init_node('start_explore_node')
    client = actionlib.SimpleActionClient('/explore_server', ExploreTaskAction)
    client.wait_for_server()

    goal = ExploreTaskGoal()
    goal.explore_boundary.header.frame_id = 'map'
    goal.explore_center.header.frame_id = 'map'
    goal.explore_center.point.x = 0.0
    goal.explore_center.point.y = 0.0
    goal.explore_center.point.z = 0.0

    client.send_goal(goal)
    client.wait_for_result()
    print('it did the thing!')
