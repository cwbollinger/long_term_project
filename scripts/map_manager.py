#!/usr/bin/env python

import os
import os.path as path
import subprocess

import rospy
from long_term_deployment.srv import RequestMap


class MapManager:

    def __init__(self):
        self.mapdir = path.normpath(path.join(path.dirname(__file__), '../maps'))
        self.served_maps = {}
        self.serve_map_service = rospy.Service('~serve_map', RequestMap, self.serve_map)
        self.end_serve_map_service = rospy.Service('~end_serve_map', RequestMap, self.end_serve_map)

    def end_serve_map(self, req):
        p = self.served_maps[req.map_name]

        if p.poll() is None: # Map Server is still running
            p.kill() # close it

        return True

    def serve_map(self, req):
        d = dict(os.environ)
        d["ROS_NAMESPACE"] = 'maps/{}'.format(req.map_name)
        map_file = path.join(self.mapdir, '{}.yaml'.format(req.map_name))
        cmdlist = ['rosrun', 'map_server', 'map_server', map_file]
        self.served_maps[req.map_name] = subprocess.Popen(cmdlist)

        return True

    def main(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == "__main__":
    rospy.init_node('map_manager')
    map_manager = MapManager()
    map_manager.main()
