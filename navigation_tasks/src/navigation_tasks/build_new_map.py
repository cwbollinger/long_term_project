import os
import subprocess

import rospy
import actionlib

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from long_term_deployment.srv import RequestMap

def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed '''
    map_name = args[0]
    rospy.loginfo('Saving map : "{}"'.format(map_name))

    expose_service = rospy.ServiceProxy('/rosduct/expose_remote_service', ROSDuctConnection)
    expose_topic = rospy.ServiceProxy('/rosduct/expose_local_topic', ROSDuctConnection)

    expose_service(conn_name='/map_manager/save_map',
                   conn_type='long_term_deployment/RequestMap',
                   alias_name='/map_manager/save_map')

    expose_topic(conn_name='/mapping/map',
                 conn_type='nav_msgs/OccupancyGrid',
                 alias_name='/map_server/newmap/{}'.format(map_name),
                 latch=True)

    r = rospy.Rate(1)
    while not stop_event.isSet():
        r.sleep()
    
    save_map = rospy.ServiceProxy('/map_manager/save_map', RequestMap)
    save_map(map_name)
    
    # close all the new tunnels we opened in the bridge
    close_service = rospy.ServiceProxy('/rosduct/close_remote_service', ROSDuctConnection)
    close_topic = rospy.ServiceProxy('/rosduct/close_local_topic', ROSDuctConnection)

    close_service(conn_name='/map_manager/save_map')
    close_topic(conn_name='/maps/newmap/{}'.format(map_name))
