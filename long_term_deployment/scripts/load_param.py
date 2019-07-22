#! /usr/bin/env python
import sys
import os
import rospy
import rospkg

try:
    print(rospy.get_param(sys.argv[1]))
except KeyError:
    print('')

