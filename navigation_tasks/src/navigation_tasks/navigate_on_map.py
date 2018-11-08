import os
import subprocess

import rospy
import actionlib

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from long_term_deployment.srv import RequestMap

def main(stop_event, args):
    ''' Takes a threading.Event to know if preemption is needed '''
    map_name = args[0]
    rospy.loginfo(map_name)

    expose_service = rospy.ServiceProxy('/rosduct/expose_remote_service', ROSDuctConnection)
    expose_topic = rospy.ServiceProxy('/rosduct/expose_remote_topic', ROSDuctConnection)

    expose_service(conn_name='/map_manager/serve_map',
                   conn_type='long_term_deployment/RequestMap',
                   alias_name='/map_manager/serve_map')

    print('request map')
    request_map = rospy.ServiceProxy('/map_manager/serve_map', RequestMap)
    request_map(map_name)
    
    print('expose map metadata')
    expose_topic(conn_name='/maps/{}/map_metadata'.format(map_name),
                 conn_type='nav_msgs/MapMetaData',
                 alias_name='/map_metadata',
                 latch=True)

    print('expose map')
    expose_topic(conn_name='/maps/{}/map'.format(map_name),
                 conn_type='nav_msgs/OccupancyGrid',
                 alias_name='/map',
                 latch=True)

    print('expose static map')
    expose_service(conn_name='/maps/{}/static_map'.format(map_name),
                   conn_type='nav_msgs/GetMap',
                   alias_name='/static_map')

    r = rospy.Rate(10)
    while not stop_event.isSet():
        r.sleep()
    
    print('closing all things')
    # close all the new tunnels we opened in the bridge
    close_service = rospy.ServiceProxy('/rosduct/close_remote_service', ROSDuctConnection)
    close_topic = rospy.ServiceProxy('/rosduct/close_remote_topic', ROSDuctConnection)

    close_service(conn_name='/map_manager/serve_map')
    close_topic(conn_name='/maps/{}/map_metadata'.format(map_name))
    close_topic(conn_name='/maps/{}/map'.format(map_name))
    close_service(conn_name='/maps/{}/static_map'.format(map_name))
