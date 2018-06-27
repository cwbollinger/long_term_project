#!/usr/bin/env python

import sys
import os
import argparse
import shutil
import time

import numpy as np

from cardboard_detection_task.srv import *
import rospy
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError

import torch
import pretrainedmodels
import pretrainedmodels.utils as utils


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from PIL import Image
import io


# def image_loader(image_name, loader):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     image = loader(image).float()
#     image = torch.autograd.Variable(image, requires_grad=False)
#     # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU


class Node:

    def __init__(self, model, loader):
        self.bridge = CvBridge()
        self.model = model
        self.loader = loader

    def handle_cardboard_query(self, req):
        # print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))

        img = self.convert(req)
        # img = image_loader(img_file, self.loader)
        img = img.unsqueeze(0)
        #rospy.logwarn(type(img))
        #rospy.logwarn(img.shape)

        output, bn_features, features_no_bn = self.model(img)

        #rospy.logwarn("Made it to here!")

        response = list(output.data.cpu().numpy()[0])

        rospy.logwarn('Result: {}'.format(response))

        return CardboardQueryResponse(response[0], response[1])

    def convert(self, req):
        rospy.logwarn(req.img)
        cv_img = self.bridge.compressed_imgmsg_to_cv2(req.img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        image = self.loader(img).float()
        image = torch.autograd.Variable(image, requires_grad=False)
        # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image.cuda()  #assumes that you're using GPU


def cardboard_query_server(model_load_path):

    evaluate = True
    do_train = False
    model_load_path = "/home/olorin/weights/cardboard_best_ckpt.pth"
    freeze_weights = True
    batch_size = 1
    workers = 4
    num_classes = 2
    arch = 'inceptionresnetv2_variant'
    print_freq = 10


    # inceptionresnetv2
    model_name = 'inceptionresnetv2_variant'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained='other', weights_in=model_load_path, freeze_weights=freeze_weights, load_imagenet=False)

    criterion = nn.CrossEntropyLoss().cuda()


    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

    loader = transforms.Compose([transforms.RandomResizedCrop(299), transforms.ToTensor(), normalize])
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    node = Node(model, loader)


    rospy.init_node('cardboard_query_server')
    s = rospy.Service('cardboard_query', CardboardQuery, node.handle_cardboard_query)
    rospy.loginfo("Ready to answer cardboard queries.")
    rospy.spin()


if __name__ == "__main__":
    #model_load_path = rospy.get_param("model_load_path")
    cardboard_query_server(None) #hard coded model path for now
