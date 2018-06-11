#! /usr/bin/env python
import os
import rospy
import rospkg


current_path = os.path.abspath(__file__)
pkg_name = rospkg.get_package_name(current_path)
ws_name = current_path.split('src/{}'.format(pkg_name))[0]
ws_name = os.path.split(ws_name[:-1])[1]
print(ws_name)

