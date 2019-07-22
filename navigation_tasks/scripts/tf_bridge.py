#! /usr/bin/env python

import math
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

def main():

    pub_tf = rospy.Publisher("/mapping/tf", TFMessage, queue_size=100)

    def tf_cb(msg):
        new_tf = []
        for transform in msg.transforms:
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                new_tf.append(transform)
        msg.transforms = new_tf
        pub_tf.publish(msg)

    tf_sub = rospy.Subscriber('/tf', TFMessage, tf_cb)

    #tfBuffer = tf2_ros.Buffer()
    #listener = tf2_ros.TransformListener(tfBuffer)
    #t = TransformStamped()

    #rate = rospy.Rate(100.0)
    #while not rospy.is_shutdown():

    #    try:
    #        trans1 = tfBuffer.lookup_transform('base_link', 'odom', rospy.Time())
    #        #trans2 = tfBuffer.lookup_transform('base_link', 'base_laser_link', rospy.Time())
    #        #theory: we don't need this because it's on the static topic?
    #    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #        rate.sleep()
    #        continue

    #    pub_tf.publish(TFMessage([trans1]))
    #    rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf_bridge')
    main()
