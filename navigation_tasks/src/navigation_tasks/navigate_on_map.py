import os
import subprocess

import rospy
import actionlib

from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion

from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from rosduct.srv import ROSDuctConnection
from long_term_deployment.srv import RequestMap

from tf.transformations import quaternion_from_euler


def main(stop_event, args, client_params):
    ''' Takes a threading.Event to know if preemption is needed '''
    if len(args) == 0:
        map_name = 'default'
    else:
        map_name = args[0]
    rospy.loginfo(map_name)

    expose_service = rospy.ServiceProxy('/rosduct/expose_remote_service', ROSDuctConnection)
    expose_topic = rospy.ServiceProxy('/rosduct/expose_remote_topic', ROSDuctConnection)
    expose_local_service = rospy.ServiceProxy('/rosduct/expose_local_service', ROSDuctConnection)

    expose_service(conn_name='/map_manager/serve_map',
                   conn_type='long_term_deployment/RequestMap',
                   alias_name='/map_manager/serve_map')

    # rospy.loginfo('request map')
    request_map = rospy.ServiceProxy('/map_manager/serve_map', RequestMap)
    request_map(map_name)

    # rospy.loginfo('expose map metadata')
    expose_topic(conn_name='/maps/{}/map_metadata'.format(map_name),
                 conn_type='nav_msgs/MapMetaData',
                 alias_name='/map_metadata',
                 latch=True)

    # rospy.loginfo('expose map')
    expose_topic(conn_name='/maps/{}/map'.format(map_name),
                 conn_type='nav_msgs/OccupancyGrid',
                 alias_name='/map',
                 latch=True)

    # rospy.loginfo('expose static map')
    expose_service(conn_name='/maps/{}/static_map'.format(map_name),
                   conn_type='nav_msgs/GetMap',
                   alias_name='/static_map')

    agent_name = rospy.get_param('~agent_name', 'default')
    expose_local_service(conn_name='/move_base/make_plan',
                         conn_type='nav_msgs/GetPlan',
                         alias_name='/{}/move_base/make_plan'.format(agent_name))

    r = rospy.Rate(10)
    localized = False
    pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped)
    start_pose = PoseWithCovarianceStamped()
    start_pose.pose_cov.pose.position.x = client_params['start_x']
    start_pose.pose_cov.pose.position.y = client_params['start_y']
    start_pose.pose_cov.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, client_params['start_a']))
    while not stop_event.isSet():
        if not localized and pose_pub.get_num_connection() > 0:
            pose_pub.publish()
            localized = True
            
        r.sleep()

    # close all the new tunnels we opened in the bridge
    # rospy.loginfo('closing navigation topics/services')
    close_service = rospy.ServiceProxy('/rosduct/close_remote_service', ROSDuctConnection)
    close_local_service = rospy.ServiceProxy('/rosduct/close_local_service', ROSDuctConnection)
    close_topic = rospy.ServiceProxy('/rosduct/close_remote_topic', ROSDuctConnection)

    close_local_service(conn_name='/move_base/make_plan')
    close_service(conn_name='/map_manager/serve_map')
    close_topic(conn_name='/maps/{}/map_metadata'.format(map_name))
    close_topic(conn_name='/maps/{}/map'.format(map_name))
    close_service(conn_name='/maps/{}/static_map'.format(map_name))
