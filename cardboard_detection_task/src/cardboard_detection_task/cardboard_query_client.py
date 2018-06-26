#!/usr/bin/env python

import sys

import rospy
from cardboard_detection_task.srv import *
from sensor_msgs.msg import CompressedImage


class Node:

    def __init__(self, image_topic):
        self.img = None
        self.called = False
        self.finished = False
        self.sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback)

    def image_callback(self, img):
        self.img = img
        if not self.called:
            self.called = True
            self.cardboard_query_client(img)
            self.sub.unregister()
            self.finished = True

    def get_img(self):
        return self.img


    def cardboard_query_client(self, img):
        rospy.wait_for_service('cardboard_query')
        try:
            cardboard_query = rospy.ServiceProxy('cardboard_query', CardboardQuery)
            resp1 = cardboard_query(img)
            self.y = resp1.a
            self.n = resp1.b
            rospy.logwarn("Cardboard: " + str(resp1.a) + "\tNothing: " + str(resp1.b))
        except rospy.ServiceException, e:
            rospy.loginfo("Service call failed: %s"%e)


def main(stop_event, args):
    image_topic = "/head_camera/rgb/image_raw/compressed"

    node = Node(image_topic)

    r = rospy.Rate(1)
    while not rospy.is_shutdown() and not node.finished:
        r.sleep()

    return [node.y, node.n]


if __name__ == "__main__":
    rospy.init_node('cardboard_query_client', anonymous=True)
    main()

    #img = node.get_img()
    #cardboard_query_client(img)
