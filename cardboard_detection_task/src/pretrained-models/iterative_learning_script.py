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
import time

from PIL import Image
from random import shuffle
import random
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids
import csv
import numpy as np



def image_loader(image_name, loader):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=False)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU  


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer.param_groups, lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def categorical_accuracy(output, target):

    batch_size = output.size(0)
    acc = 0.0
    for i in range(batch_size):
        output_instance = output[i]
        target_instance = target[i]

        output_argmax = output_instance.argmax(dim=-1)
        acc += float(output_argmax.tolist() == target_instance.tolist())

    acc = acc/batch_size

    return acc



def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # acc_meter = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # # compute output
        # output = model(input_var)
        # loss = criterion(output, target_var)

        # switch to eval mode
        model.eval()

        # compute output
        output, output2, output3 = model(input)
        loss = criterion(output, target)

        # batch norm? -- TODO

        # acc = categorical_accuracy(output.data, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # acc_meter.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))



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

            output, output2, output3 = model(input)
            loss = criterion(output, target)

            # acc = categorical_accuracy(output.data, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 2))
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

    evaluate = False
    do_train = True
    model_path = "/home/scatha/research_ws/src/lifelong_object_learning/src/pretrained-models/weights/"
    model_load_name = "imagenet_imagenet_frozen_rms_b128_lr0.001_e3.pth"
    # model_load_name = "flickr_frozen.pth"
    # model_save_name = "imagenet.pth"
    model_save_name = "iterative_rgbd_16_lr_0.001_e3.pth"
    model_save_name_ckpt = "iterative_rgbd_16_lr_0.001_e3_best_ckpt.pth"
    freeze_weights = True
    # data = None
    # traindir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train"
    # valdir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/scraped/train"
    traindir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train"
    valdir = "/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train"
    # batch_size = 1256
    batch_size = 128
    workers = 4
    num_classes = 6

    # # SGD params
    # lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    # # Adadelta params
    # lr = 1.0
    # rho = 0.9
    # eps = 1e-6
    # weight_decay = 0

    lr = 0.001

    start_epoch = 0
    # epochs = 30
    epochs = 1
    # epochs = 2
    # epochs = 2
    arch = 'inceptionresnetv2_variant'
    # print_freq = 10
    print_freq = 30

    early_stopping = False
    patience = 0


    train_data_dir_base = "/media/scatha/Data/lifelong_object_learning/training_data/rgbd-iterative/"
    subset_dir = "/media/scatha/Data/lifelong_object_learning/training_data/subset_selected"
    temp_dir = "/media/scatha/Data/lifelong_object_learning/training_data/temp/"
    epochs = 3
    batch_size = 16
    es_min_delta = 1e-1
    patience_val = 0
    subset_size_per_class = 32
    num_features = 3
    dist_metric = 'l2'
    RANDOM_CHOICE = False


    # # load from imagenet
    # model_name = 'inceptionresnetv2_variant'
    # model = pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained='imagenet+background', weights_in=model_path + model_load_name, freeze_weights=freeze_weights)


    # inceptionresnetv2
    model_name = 'inceptionresnetv2_variant'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained='other', weights_in=model_path + model_load_name, freeze_weights=freeze_weights, load_imagenet=True)

    model = torch.nn.DataParallel(model).cuda()

    # # nasnetalarge
    # model_name = 'nasnetalarge'
    # model = pretrainedmodels.__dict__[model_name](num_classes=1001, pretrained='imagenet+background')


    ## print weights
    # print (model.last_linear.weight.shape)
    # print (model.last_linear.weight)


    # Data loading code
    # traindir = os.path.join(data, 'train')
    # valdir = os.path.join(data, 'val')

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(max(model.input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)



    # if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
    #     scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
    # else:
    #     scale = 0.875
    # scale = 0.875

    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))

    # val_tf = pretrainedmodels.utils.TransformImage(model, scale=scale)
    # train_tf = pretrainedmodels.utils.TransformImage(model, scale=scale)



    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, train_tf),
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True)


    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, val_tf),
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True)


    train_sampler = None

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])

    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                              std=[0.5, 0.5, 0.5])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(224),
    #         transforms.Resize(299),
    #         transforms.RandomResizedCrop(299),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(224),
    #         transforms.Resize(299),
    #         transforms.RandomResizedCrop(299),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))



    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #     num_workers=workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True)



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    ## Optimizer
    # SGC
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
    #                             lr=lr,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)

    # Adadelta
    # optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), 
    #                             lr=lr,
    #                             rho=rho,
    #                             eps = eps,
    #                             weight_decay=weight_decay)

    # RMSprop
    # RMSprop
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr,
                                alpha=0.9,
                                eps = 1e-08,
                                weight_decay=0,
                                momentum=0,
                                centered=False)


    if evaluate:
        validate(val_loader, model, criterion, print_freq)
        return







    object_classes = ["bowl", "calculator", "cell_phone", "coffee_mug", "notebook", "plate"]
    # object_classes = ["bowl"]
    num_instances = 4

    object_datasets = []
    for object_class in object_classes:
        for i in range(num_instances):
            instance = i + 1
            dataset = object_class + "_" + str(instance)
            object_datasets.append(dataset)
    shuffle(object_datasets)



    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])

    loader = transforms.Compose([transforms.Resize(299), transforms.RandomResizedCrop(299), transforms.ToTensor(), normalize])



    for dataset in object_datasets:

        print ("Dataset : " + dataset)


        traindir = train_data_dir_base + dataset
        valdir = train_data_dir_base + dataset


        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(299),
                transforms.RandomResizedCrop(299),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(299),
                transforms.RandomResizedCrop(299),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))



        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)



        # Train
        t0 = time.time()
        if do_train:
            best_prec, prec5 = validate(val_loader, model, criterion, print_freq)
            early_stopping_buffer = []
            early_stopping_buffer.append(best_prec)
            for epoch in range(start_epoch, epochs):

                # print ()
                # print ("epoch: " + str(epoch) + "/" + str(epochs))

                # potentially implemented in optimizer?
                # optimizer.param_groups, lr = adjust_learning_rate(optimizer, epoch, lr)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, print_freq)

                # evaluate on validation set
                prec1, prec5 = validate(val_loader, model, criterion, print_freq)
                # print (prec1)

                # early stopping
                if early_stopping:
                    if len(early_stopping_buffer) == (patience+1):

                        better = True
                        for prec in early_stopping_buffer:
                            if prec1 < prec:
                                better = False
                        if better:
                            print ("Stopping")
                            break

                        early_stopping_buffer.pop(0)
                        early_stopping_buffer.append(prec1)

                    else:
                        early_stopping_buffer.append(prec1)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec
                best_prec = max(prec1, best_prec)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec,
                }, is_best)


            # # save model_name
            # # checkpoint? TODO!
            # print ("Saving model")
            # torch.save(model.state_dict(), model_path + model_save_name)

            # best_checkpoint = torch.load('model_best.pth.tar')
            # model.load_state_dict(best_checkpoint['state_dict'])
            # torch.save(model.state_dict(), model_path + model_save_name_ckpt)

            # # time
            # final_time = time.time() - t0
            # print ("Time: " + str(final_time))



        ## update subset
        print ("Updating subset")

        object_class = dataset[:-2]
        object_class_index = None
        for i in range(len(object_classes)):
            if object_class == object_classes[i]:
                object_class_index = i

        # move new images into class subset
        source_dir = traindir + "/" + object_class
        dest_dir = subset_dir + "/" + object_class
        for file in os.listdir(source_dir):
            source_file = source_dir + "/" + file
            dest_file = dest_dir + "/" + file
            shutil.copy(source_file, dest_file)

        # get activations for each subset image
        weights = model.module.last_linear.weight
        class_weights = weights.data.cpu().numpy()[object_class_index]
        class_weights = list(class_weights)

        # layer = model.layers[-1]   
        # weights = layer.get_weights()[0]
        # weights_by_class = np.swapaxes(weights, 0, 1)
        # class_weights = weights_by_class[object_class_index]
        # print (class_weights.shape)
        # class_weights = list(class_weights)
        # print (len(class_weights))


        # get top num_features weight indices
        weight_tuples = []
        for i in range(len(class_weights)):
            t = [abs(class_weights[i]), i]
            weight_tuples.append(t)


        sorted_weight_tuples = sorted(weight_tuples, key=lambda tup: tup[0])
        sorted_weight_tuples = list(reversed(sorted_weight_tuples))
        top_weight_indices = []
        top_weights = []
        for i in range(num_features):
            weight = sorted_weight_tuples[i][0]
            weight_index = sorted_weight_tuples[i][1]
            top_weight_indices.append(weight_index)
            top_weights.append(weight)

        # get values
        values = []

        model.eval()
        files = os.listdir(dest_dir)
        for file in files:
            if file.endswith(".png"):


                img_file = os.path.join(dest_dir, file)

                img = image_loader(img_file, loader)
                img = img.unsqueeze(0)


                output, bn_features, features_no_bn = model(img)

                layer_output = bn_features.data.cpu().numpy()
                layer_output = list(layer_output[0])


                # img = cv2.imread(img_file)
                # img = cv2.resize(img, (299, 299))
                # img = np.expand_dims(img, axis=0)

                # # output in test mode = 0
                # layer_output = get_2nd_to_last_layer_output([img, 0])[0][0]
                # # print(layer_output)
                # # layer_output_other = get_3rd_to_last_layer_output([img, 0])[0][0]
                # # print(layer_output_other)
                # # print ()
                # layer_output = list(layer_output)

                value = []
                for index in top_weight_indices:
                    output = layer_output[index]
                    value.append(output)
                values.append(value)
        values = np.array(values)
        num_vals = values.shape[0]


        # calculate distance matrix
        D = pairwise_distances(values, metric=dist_metric)

        # # weighted distance matrix
        # norms = np.linalg.norm(values, axis=1)
        # D = np.zeros((num_vals, num_vals))
        # for i in range(num_vals -1):
        #     for j in range(num_vals - i -1):
        #         k = j + i + 1
        #         distance = calc_distance(i, k, values, norms, top_weights, metric=dist_metric)
        #         D[i][k] = distance
        #         D[k][i] = distance
          


        if RANDOM_CHOICE:
            # random choice
            indices = list(range(num_vals))
            M = []
            for i in range(subset_size_per_class):
                index = random.choice(indices)
                M.append(index)
                indices.remove(index)

        else:
            # use k medoids to get representative images (medoids) 
            print ("Applying K Medoids") 
            t0 = time.time()  
            M, C = kmedoids.kMedoids(D, subset_size_per_class)
            final_time = time.time() - t0
            print ("K medoids time: " + str(final_time))


        # copy new subset to temp
        files_to_keep = []
        for index in M:
            file = files[index]
            files_to_keep.append(file)
            subset_filepath = dest_dir + "/" + file
            temp_filepath = temp_dir + file
            shutil.copyfile(subset_filepath,temp_filepath)

        # empty old class subset
        shutil.rmtree(dest_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # copy new subset from temp
        for file in files_to_keep:
            temp_filepath = temp_dir + file
            subset_filepath = dest_dir + "/" + file
            shutil.copyfile(temp_filepath, subset_filepath)

        # empty temp
        shutil.rmtree(temp_dir)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)



        ## retrain on subset

        print ("Retraining on subset")
        t0 = time.time() 

        # train_data_dir = subset_dir

        train_dataset = datasets.ImageFolder(
            subset_dir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(299),
                transforms.RandomResizedCrop(299),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            subset_dir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(299),
                transforms.RandomResizedCrop(299),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))



        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)



        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(train_data_dir, train_tf),
        #     batch_size=batch_size, shuffle=False,
        #     num_workers=workers, pin_memory=True)


        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(train_data_dir, val_tf),
        #     batch_size=batch_size, shuffle=False,
        #     num_workers=workers, pin_memory=True)

        if do_train:
            best_prec, prec5 = validate(val_loader, model, criterion, print_freq)
            early_stopping_buffer = []
            early_stopping_buffer.append(best_prec)
            for epoch in range(start_epoch, epochs):

                # print ()
                # print ("epoch: " + str(epoch) + "/" + str(epochs))

                # potentially implemented in optimizer?
                # optimizer.param_groups, lr = adjust_learning_rate(optimizer, epoch, lr)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, print_freq)

                # evaluate on validation set
                prec1, prec5 = validate(val_loader, model, criterion, print_freq)
                # print (prec1)

                # early stopping
                if early_stopping:
                    if len(early_stopping_buffer) == (patience+1):

                        better = True
                        for prec in early_stopping_buffer:
                            if prec1 < prec:
                                better = False
                        if better:
                            print ("Stopping")
                            break

                        early_stopping_buffer.pop(0)
                        early_stopping_buffer.append(prec1)

                    else:
                        early_stopping_buffer.append(prec1)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec
                best_prec = max(prec1, best_prec)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec,
                }, is_best)


        final_time = time.time() - t0
        print ("Dataset train time: " + str(final_time))


    # save model_name
    # checkpoint? TODO!
    print ("Saving model")
    torch.save(model.state_dict(), model_path + model_save_name)





if __name__ == "__main__":
    main()
