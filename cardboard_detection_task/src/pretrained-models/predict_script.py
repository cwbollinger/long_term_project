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
from random import shuffle
from sklearn.metrics.pairwise import pairwise_distances
import csv



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # print (output[0])
    # print (target.shape)
    # print ()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def image_loader(image_name, loader):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=False)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU  



def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # acc_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            # input = input.cuda(non_blocking=True)

            # input_var = torch.autograd.Variable(input, requires_grad=False)
            # target_var = torch.autograd.Variable(target, requires_grad=False)

            # # compute output
            # output = model(input_var)
            # loss = criterion(output, target_var)

            output, outpu2, output3 = model(input)
            loss = criterion(output, target)

            # acc = categorical_accuracy(output.data, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # acc_meter.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      # 'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg



def main():

    evaluate = True
    do_train = False
    model_path = "/home/scatha/research_ws/src/lifelong_object_learning/src/pretrained-models/weights/"
    model_load_name = "imagenet_rgbd_frozen_rms_b128_lr0.001_e60.pth"
    # model_save_name = "imagenet.pth"
    freeze_weights = True
    # data = None
    traindir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train"
    valdir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train"
    # batch_size = 1256
    batch_size = 128
    workers = 4
    num_classes = 6

    # # SGD params
    # lr = 0.1
    # momentum = 0.9
    # weight_decay = 1e-4

    # Adadelta params
    lr = 1.0
    rho = 0.9
    eps = 1e-6
    weight_decay = 0

    start_epoch = 0
    epochs = 90
    # epochs = 2
    arch = 'inceptionresnetv2_variant'
    # print_freq = 10
    print_freq = 10


    # # load from imagenet
    # model_name = 'inceptionresnetv2_variant'
    # model = pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained='imagenet+background', weights_in=model_path + model_load_name, freeze_weights=freeze_weights)


    # inceptionresnetv2
    model_name = 'inceptionresnetv2_variant'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained='other', weights_in=model_path + model_load_name, freeze_weights=freeze_weights)

    # model = torch.nn.DataParallel(model).cuda()



    # load_img = utils.LoadImage()
    # scale = 0.875

    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))

    # tf_img = pretrainedmodels.utils.TransformImage(model, scale=scale)


    # # test batch norm
    # # files_dirs = ["/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/bowl",
    # #             "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/calculator",
    # #             "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/cell_phone",
    # #             "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/coffee_mug",
    # #             "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/notebook",
    # #             "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/plate"]

    # files_dirs = ["/home/scatha/research_ws/src/lifelong_object_learning/data/temp"]

    # for files_dir_index in range(len(files_dirs)): 
    #     files_dir = files_dirs[files_dir_index]   
    #     values = []

    #     files = os.listdir(files_dir)
    #     for file in files:
    #         if file.endswith(".png"):
    #             path_img = os.path.join(files_dir, file)

    #             input_img = load_img(path_img)
    #             input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
    #             input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    #             input_vals = torch.autograd.Variable(input_tensor, requires_grad=False)


    #             output_features = model.features(input_vals) # 1x14x14x2048 size may differ
    #             # output_logits = model.logits(output_features) # 1x1000
    #             # print (output_features)\

    #             output_features = output_features.data.numpy()
    #             values.append(list(output_features))

    #     values = np.array(values)

    #     print (values.shape)
        
    #     print ("Index: " + str(files_dir_index))
    #     print ("mean")
    #     print (np.mean(values, axis=0))
    #     print ("std")
    #     print (np.std(values, axis=0))





    # scale = 0.875

    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))


    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(224),
    #         transforms.RandomResizedCrop(299),
    #         # transforms.Resize(299),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    # for i in val_dataset:
    #     print (i[0])


    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True)



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    # switch to eval mode
    model.eval()

    # if evaluate:
    #     validate(val_loader, model, criterion, print_freq)
    #     return






    ##################################

    ## TEST BATCH NORM

    ##################################


    # files_dirs = ["/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/bowl",
    #                 "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/calculator",
    #                 "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/cell_phone",
    #                 "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/coffee_mug",
    #                 "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/notebook",
    #                 "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train/plate"]


    files_dirs = ["/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/bowl",
                "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/calculator",
                "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/cell_phone",
                "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/coffee_mug",
                "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/notebook",
                "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train/plate"]



    # print(model._modules)
    # last_bn_layer = model._modules.get('last_bn')
    # avgpool_1a_layer = model._modules.get('avgpool_1a')

    # print(last_bn_layer)
    # print(avgpool_1a_layer)


    for files_dir_index in range(len(files_dirs)): 
        files_dir = files_dirs[files_dir_index]   
        values = []
        values_other = [] 

        loader = transforms.Compose([transforms.RandomResizedCrop(299), transforms.ToTensor(), normalize])

        files = os.listdir(files_dir)
        for file in files:
            if file.endswith(".png"):
                img_file = os.path.join(files_dir, file)

                img = image_loader(img_file, loader)
                img = img.unsqueeze(0)


                output, bn_features, features_no_bn = model(img)
                # output = model(img)


                # bn_features = torch.zeros(1536)
                # features_no_bn = torch.zeros(1536)

                # def copy_data_1(m, i, o):
                #     bn_features.copy_(o.data)

                # def copy_data_2(m, i, o):
                #     features_no_bn.copy_(o.data)

                # h1 = last_bn_layer.register_forward_hook(copy_data_1)
                # h2 = avgpool_1a_layer.register_forward_hook(copy_data_2)

                # model(img)

                # h1.remove()
                # h2.remove()


                # features = model.features(img)
                # bn_features = model.features_avgpool(features)

                # features_no_bn = model.features_avgpool_no_bn(features)

                # img = cv2.imread(img_file)
                # img = cv2.resize(img, (299, 299))
                # img = np.expand_dims(img, axis=0)

                # # output in test mode = 0
                # layer_output = get_2nd_to_last_layer_output([img, 0])[0][0]
                # # print(layer_output)
                # layer_output_other = get_3rd_to_last_layer_output([img, 0])[0][0]
                # # print(layer_output_other)
                # # print ()

                layer_output = bn_features.data.cpu().numpy()
                layer_output = list(layer_output)
                values.append(layer_output)

                layer_output_other = features_no_bn.data.cpu().numpy()
                layer_output_other = list(layer_output_other)
                values_other.append(layer_output_other)

        values = np.array(values)
        values_other = np.array(values_other)
        
        print ("Index: " + str(files_dir_index))
        print ("mean")
        print (np.mean(values, axis=0))
        print ("std")
        print (np.std(values, axis=0))

        print ()
        print ("mean")
        print (np.mean(values_other, axis=0))
        print ("std")
        print (np.std(values_other, axis=0))
        print ()
        print ()






    ###############################################################


    # img_file  = "/home/scatha/research_ws/src/lifelong_object_learning/data/rgbd-dataset/bowl/bowl_1/bowl_1_1_1_crop.png"

    # loader = transforms.Compose([transforms.RandomResizedCrop(299), transforms.ToTensor(), normalize])
    # model = torch.nn.DataParallel(model).cuda()
    # model.eval()

    # img = image_loader(img_file, loader)
    # img = img.unsqueeze(0)

    # output, bn_features, features_no_bn = model(img)
    # print (list(output.data.cpu().numpy()[0]))


    ##################################

    ## DISTANCE COMPUTATION

    ##################################

    # num_features = 3
    # dist_metric = 'l2'


    # object_classes = ["bowl", "calculator", "cell_phone", "coffee_mug", "notebook", "plate"]
    # # object_classes = ["bowl"]
    # num_instances = 4

    # datasets = []
    # for object_class in object_classes:
    #     for i in range(num_instances):
    #         instance = i + 1
    #         dataset = object_class + "_" + str(instance)
    #         datasets.append(dataset)
    # shuffle(datasets)



    # object_class = "bowl"

    # object_class_index = None
    # for i in range(len(object_classes)):
    #     if object_class == object_classes[i]:
    #         object_class_index = i


    # # get activations for each subset image

    # weights = model.last_linear.weight
    # class_weights = weights.data.numpy()[object_class_index]
    # class_weights = list(class_weights)

    # # layer = model.layers[-1]   
    # # weights = layer.get_weights()[0]
    # # weights_by_class = np.swapaxes(weights, 0, 1)
    # # class_weights = weights_by_class[object_class_index]
    # # class_weights = list(class_weights)

    # # get top num_features weight indices
    # weight_tuples = []
    # for i in range(len(class_weights)):
    #     t = [class_weights[i], i]
    #     weight_tuples.append(t)

    # sorted_weight_tuples = sorted(weight_tuples, key=lambda tup: tup[0])
    # sorted_weight_tuples = list(reversed(sorted_weight_tuples))
    # top_weight_indices = []
    # top_weights = []
    # for i in range(num_features):
    #     weight = sorted_weight_tuples[i][0]
    #     weight_index = sorted_weight_tuples[i][1]
    #     top_weight_indices.append(weight_index)
    #     top_weights.append(weight)

    # # get values
    # # source_dir = train_data_dir + "/" + object_class
    # # for file in os.listdir(source_dir):
    # #     source_file = source_dir + "/" + file
    # values = []

    # direc = "/home/scatha/research_ws/src/lifelong_object_learning/data/rgbd-dataset/"
    # files = ["bowl/bowl_1/bowl_1_1_1_crop.png", "bowl/bowl_1/bowl_1_1_2_crop.png", "bowl/bowl_1/bowl_1_2_1_crop.png", "bowl/bowl_1/bowl_1_4_1_crop.png", 
    #         "bowl/bowl_2/bowl_2_1_1_crop.png", "bowl/bowl_3/bowl_3_1_1_crop.png", "bowl/bowl_4/bowl_4_1_1_crop.png", "bowl/bowl_5/bowl_5_1_1_crop.png",
    #         "calculator/calculator_1/calculator_1_1_1_crop.png", "cell_phone/cell_phone_1/cell_phone_1_1_1_crop.png", "coffee_mug/coffee_mug_1/coffee_mug_1_1_1_crop.png", 
    #         "notebook/notebook_1/notebook_1_1_1_crop.png", "plate/plate_1/plate_1_1_1_crop.png"]

    # loader = transforms.Compose([transforms.RandomResizedCrop(299), transforms.ToTensor(), normalize])
    # model = torch.nn.DataParallel(model).cuda()
    # model.eval()

    # for file in files:
    #     img_file = os.path.join(direc, file)

    #     img = image_loader(img_file, loader)
    #     img = img.unsqueeze(0)

    #     output, bn_features, features_no_bn = model(img)

    #     # img = cv2.imread(img_file)
    #     # img = cv2.resize(img, (299, 299))
    #     # img = np.expand_dims(img, axis=0)

    #     # output in test mode = 0
    #     # layer_output = get_2nd_to_last_layer_output([img, 0])[0][0]
    #     layer_output = bn_features.data.cpu().numpy()
    #     layer_output = list(layer_output[0])
    #     value = []
    #     for index in top_weight_indices:
    #         output = layer_output[index]
    #         weighted_output = class_weights[index]*output

    #         value.append(output)
    #     values.append(value)
    # values = np.array(values)
    # num_vals = values.shape[0]

    # D = pairwise_distances(values, metric=dist_metric)
    # # print (D)
    # # print ()


    # rows = D.shape[0]
    # cols = D.shape[1]

    # csv_file = "/home/scatha/Desktop/res.csv"
    # with open(csv_file, 'w') as f:
    #     writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for r in range(rows):
    #         line = []
    #         for c in range(cols):
    #             line.append(str(D[r][c]))
    #         writer.writerow(line)



    #########################################

    return





if __name__ == "__main__":
    main()
