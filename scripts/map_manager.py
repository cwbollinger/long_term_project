#!/usr/bin/env python

import os
import os.path as path
import subprocess
from datetime import datetime

import rospy
from long_term_deployment.srv import RequestMap, RequestMapResponse

class MapManager:

    def __init__(self):
        self.mapdir = path.normpath(path.join(path.dirname(__file__), '../maps'))
        self.served_maps = {}
        self.serve_map_service = rospy.Service('~serve_map', RequestMap, self.serve_map)
        self.end_serve_map_service = rospy.Service('~end_serve_map', RequestMap, self.end_serve_map)
        self.save_map_service = rospy.Service('~save_map', RequestMap, self.save_map)

    def end_serve_map(self, req):
        p = self.served_maps[req.map_name]

        if p.poll() is None: # Map Server is still running
            p.kill() # close it

        del self.served_maps[req.map_name]

        return RequestMapResponse()

    def serve_map(self, req):
        if req.map_name in self.served_maps: # we are already serving this map
            return RequestMapResponse()

        # this is how you run a node using rosrun in a different namespace
        d = dict(os.environ)
        d["ROS_NAMESPACE"] = 'maps/{}'.format(req.map_name)

        # build path to map file, maps must be in long_term_server/map directory
        map_file = path.join(self.mapdir, '{}.yaml'.format(req.map_name))
        rospy.loginfo(map_file)

        # run as a subprocess
        cmdlist = ['rosrun', 'map_server', 'map_server', map_file]
        self.served_maps[req.map_name] = subprocess.Popen(cmdlist, env=d)

        return RequestMapResponse()

    def save_map(self, req):
        d = dict(os.environ)
        d["ROS_NAMESPACE"] = 'map_manager'
        t = rospy.Time().now().to_time()
        map_name = '{}_{}'.format(req.map_name, datetime.fromtimestamp(t))
        map_file = path.join(self.mapdir, map_name)
        cmdlist = ['rosrun', 'map_server', 'map_saver', '-f {}'.format(map_file), 'map:=newmap/{}'.format(req.map_name)]
        p = subprocess.Popen(cmdlist, env=d)
        r = rospy.Rate(10)
        while p.poll() is None:
            r.sleep()

        return RequestMapResponse()

    def main(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == "__main__":
    rospy.init_node('map_manager')
    map_manager = MapManager()
    map_manager.main()
