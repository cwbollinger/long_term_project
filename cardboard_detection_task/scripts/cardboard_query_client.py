#!/usr/bin/env python

import sys
import rospy
from cardboard_task.srv import *
from sensor_msgs.msg import CompressedImage



class Node:

    def __init__(self, image_topic):
        self.img = None
        rospy.Subscriber(image_topic, CompressedImage, self.image_callback)


    def image_callback(img):
        self.img = img

    def get_img():
        return self.img


def cardboard_query_client(img):
    rospy.wait_for_service('cardboard_query')
    try:
        cardboard_query = rospy.ServiceProxy('cardboard_query', CardboardQuery)
        resp1 = cardboard_query(img)
        rospy.loginfo("Cardboard: " + str(resp1.a) + "\tNothing: " + str(resp1.b))
        return
    except rospy.ServiceException, e:
        rospy.loginfo("Service call failed: %s"%e)



if __name__ == "__main__":

    image_topic = "/head_camera/rgb/image_raw/compressed"

    rospy.init_node('cardboard_query_client', anonymous=True)
    node = Node(image_topic)

    img = node.get_img()

    cardboard_query_client(img)