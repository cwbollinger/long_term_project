import os
import subprocess

import rospy
import actionlib
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from rosduct.msg import ROSDuctConnection
from long_term_deployment.srv import RequestMap

def main(stop_event, args):
    ''' Takes a threading.Event to know if preemption is needed '''
    map_name = args[0]

    expose_service = rospy.ServiceProxy('/rosduct/expose_remote_service', ROSDuctConnection)
    expose_topic = rospy.ServiceProxy('/rosduct/expose_remote_topic', ROSDuctConnection)

    expose_service('/map_manager/serve_map', 'long_term_deployment/RequestMap', '/map_manager/serve_map')

    request_map = rospy.ServiceProxy('/map_manager/serve_map', RequestMap)
    request_map(map_name)
    
    expose_topic('/maps/{}/map_metadata'.format(map_name), 'nav_msgs/MapMetaData', '/map_metadata')
    expose_topic('/maps/{}/map'.format(map_name), 'nav_msgs/OccupancyGrid', '/map')
    expose_service('/maps/{}/static_map'.format(map_name), 'nav_msgs/GetMap', '/static_map')

    r = rospy.Rate(10)
    while not stop_event.isSet():
        r.sleep()
    
