#!/usr/bin/env python

import sys
import rospy
from cardboard_detection_task.srv import *
from sensor_msgs.msg import CompressedImage



class Node:

    def __init__(self, image_topic, callback):
        self.img = None
        self.called = False
        self.finished = False
        self.once_callback = callback
        self.sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback)


    def image_callback(self, img):
        self.img = img
        if not self.called:
            self.called = True
            self.once_callback(img)
            self.sub.unregister()
            self.finished = True

    def get_img(self):
        return self.img


def cardboard_query_client(img):
    rospy.wait_for_service('cardboard_query')
    try:
        cardboard_query = rospy.ServiceProxy('cardboard_query', CardboardQuery)
        resp1 = cardboard_query(img)
        rospy.logwarn("Cardboard: " + str(resp1.a) + "\tNothing: " + str(resp1.b))
        return
    except rospy.ServiceException, e:
        rospy.loginfo("Service call failed: %s"%e)



if __name__ == "__main__":

    image_topic = "/head_camera/rgb/image_raw/compressed"

    rospy.init_node('cardboard_query_client', anonymous=True)
    node = Node(image_topic, cardboard_query_client)

    r = rospy.Rate(1)
    while not rospy.is_shutdown() and not node.finished:
        r.sleep()

    #img = node.get_img()
    #cardboard_query_client(img)
