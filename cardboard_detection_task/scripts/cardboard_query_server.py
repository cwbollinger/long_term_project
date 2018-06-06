#!/usr/bin/env python

from cardboard_task.srv import *
import rospy
from sensor_msgs.msg import CompressedImage

import torch
import pretrainedmodels
import pretrainedmodels.utils as utils

import argparse
import os
import shutil
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
import numpy as np

from PIL import Image
import io




# def image_loader(image_name, loader):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     image = loader(image).float()
#     image = torch.autograd.Variable(image, requires_grad=False)
#     # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU  


def convert(loader, req):

    img =  Image.open(io.BytesIO(bytearray(req.img.data)))
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=False)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU  



class Node:

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def handle_cardboard_query(req):
        # print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))

        img = convert(self.loader, req)
        # img = image_loader(img_file, self.loader)
        img = img.unsqueeze(0)

        output, bn_features, features_no_bn = self.model(img)

        response = list(output.data.cpu().numpy()[0])


        return CardboardQueryResponse(response[0], response[1])



def cardboard_query_server(model_load_path):

    evaluate = True
    do_train = False
    # model_load_path = "/home/scatha/research_ws/src/lifelong_object_learning/src/pretrained-models/weights/cardboard_best_ckpt.pth"
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
    model_load_path = rospy.get_param("model_load_path")
    cardboard_query_server(model_load_path)