# coding: utf-8
from __future__ import division
import numpy as np
import pdb
import math
from . import data_generators
import copy
import cv2
class_num =200
part_map_num = {'head':0,'legs':1,'wings':2,'back':3,'belly':4,'breast':5,'tail':6}
part_map_name = {}


net_size=[38,56]
#[head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]

#def get_label_from_voc(all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping):

class get_voc_label(object):
    def __init__(self,all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping,config,trainable=True):
        self.all_imgs = all_imgs
        self.classes_count = classes_count
        self.class_mapping = class_mapping
        self.bird_classes_count = bird_classes_count
        self.bird_class_mapping = bird_class_mapping
        self.max_batch = len(all_imgs)
        self.batch_index = 0
        self.epoch = 0
        self.part_num =  7#len(classes_count)
        self.bird_class_num = len(bird_classes_count)
        self.net_size = [38,56]
        self.C =config
        self.input_img_size_witdth = 600
        self.input_img_size_heigth = 600
        if self.C.network=='resnet50':
            self.get_outputsize =self.get_img_output_length_res50
        elif self.C.network=='vgg':
            self.get_outputsize = self.get_img_output_length_vgg
        else:
            raise ValueError('DSFA')
        if trainable:
            self.trainable = 'trainval'
        else:
            self.trainable = 'test'

    def get_next_batch(self):
        img = self.all_imgs[self.batch_index]
        while img['imageset']!= self.trainable:
            self.batch_index+=1
            if self.batch_index>=self.max_batch:
                self.batch_index=0
                self.epoch+=1
            img = self.all_imgs[self.batch_index]
        label = self.bird_class_mapping[img['bird_class_name']]
        boxlist =[]
        size_w = img['width']
        size_h = img['height']
        for bbox in img['bboxes']:
            outbox ={}
            outbox['name']=bbox['class']
            cor = np.zeros(4)
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1= bbox['y1']
            y2 = bbox['y2']
            h = y2-y1
            w = x2-x1
            x1/=size_w
            y1/=size_h
            h/=size_h
            w /= size_w
            cor =np.array([x1,y1,w,h])
            outbox['cor'] =cor
            boxlist.append(outbox)
        img_path = img['filepath']
        boxdict, labellist ,labelnpout=self.match(boxlist, label)
        self.batch_index += 1
        if self.batch_index >= self.max_batch:
            self.batch_index = 0
            self.epoch += 1
        return img_path,boxdict,labellist,labelnpout
    def next_batch(self,batech_size):
        img_input_np= np.zeros([batech_size,self.input_img_size_heigth,self.input_img_size_witdth,3])
        netout_width, netout_height = self.get_outputsize(width=self.input_img_size_witdth, height=self.input_img_size_heigth)
        netout_height =149
        netout_width =149
        part_roi_input = np.zeros([batech_size,self.part_num,4],dtype=np.int16)
        labellist =[]
        for nn in range(self.part_num):
            labellist.append(np.zeros([batech_size,self.bird_class_num+1]))
        for n_b in range(batech_size):
            img = self.all_imgs[self.batch_index]
            while img['imageset'] != self.trainable:
                self.batch_index += 1
                if self.batch_index >= self.max_batch:
                    self.batch_index = 0
                    self.epoch += 1
                img = self.all_imgs[self.batch_index]
            img_path = img['filepath']
            img_np = self.read_prepare_img(img_path,img['width'],img['height'],width_to_resize=self.input_img_size_witdth,heigth_to_resize=self.input_img_size_heigth)
            img_input_np[n_b,:,:,:]=img_np
            #netout_width,netout_height= self.get_outputsize(width=self.input_img_size_witdth,height=self.input_img_size_heigth)
            bird_class_label_num = self.bird_class_mapping[img['bird_class_name']]
            if 1:
                boxlist = []
                for i in range(self.part_num):
                    part_roi_input[n_b,i,:]=np.array([0,0,netout_width-1,netout_height-1],dtype=np.int16)
                    #boxlist.append(np.array([0,0,netout_width-1,netout_height-1],dtype=np.int16))
                #boxnp = np.copy(boxlist)
                #boxnp = np.expand_dims(boxnp,axis=0)
                """print boxnp.shape
                print self.part_num
                print boxnp[0,0,:]
                assert boxnp.shape==[1,self.part_num,4]
                assert boxnp[0,0,:]==np.array([0,0,netout_width,netout_height],dtype=np.int16)"""
            #boxnp = np.zeros([1, self.part_num, 4])
            check_dict = {}
            nnn = 0
            for bbox in img['bboxes']:
                nnn+=1
                part_index = self.class_mapping[bbox['class']]
                if part_index != 0:
                    continue
                if str(part_index) not in check_dict:
                    check_dict[str(part_index)] = part_index
                else:
                    raise ValueError('sdff')
                #print bbox
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1= bbox['y1']
                y2 = bbox['y2']
                w = x2-x1
                h = y2-y1
                x1= x1/img['width']*netout_width
                w = w/img['width']*netout_width
                y1 = y1/img['height']*netout_height
                h = h/img['height']*netout_height
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                part_roi_input[n_b,part_index,:] = [x1,y1,w,h]
                labellist[part_index][n_b,bird_class_label_num] = 1
                labellist[part_index][n_b,0] = 1
                for i in range(7):
                    if labellist[i][n_b,0] == 1:
                        if str(i) not in check_dict:
                            raise ValueError(str(i)+' nnn is :'+str(nnn))
            self.batch_index += 1
            if self.batch_index >= self.max_batch:
                self.batch_index = 0
                self.epoch += 1
        return img_input_np,part_roi_input,labellist,img['filepath'] #img_path,img['index']


    def match(self,boxlist, label):
        # boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间
        # label的内同是一个数
        labellist = []
        boxdict = {}
        labelnp = np.zeros([1,class_num + 1])
        for i in range(7):
            labellist.append(labelnp)
        labelnpout = np.zeros([7,class_num+1])

        if len(labellist) != 7:
            raise ValueError('SDFA')
        for box in boxlist:
            index = part_map_num[box['name']]
            labellist[index][0][0] = 1
            labellist[index][0][label+1] = 1

            labelnpout[index][0] = 1
            labelnpout[index][label+1] =1
            x = box['cor'][0]
            y = box['cor'][1]
            w = box['cor'][2]
            h = box['cor'][3]
            x *= net_size[1]
            w *= net_size[1]
            y *= net_size[0]
            h *= net_size[0]
            cor_np = np.array([x, y, w, h])
            cor_np = np.expand_dims(cor_np, axis=0)
            boxdict[box['name']] = cor_np

        npnone = np.zeros([1,1,4])
        # [head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]
        cname = ['head','legs','wings','back','belly','breast','tail']
        for onecname in cname:
            if onecname not in boxdict:
                boxdict[onecname] = npnone

        return boxdict, labellist,labelnpout
    def read_prepare_img(self,img_path,width,height,width_to_resize,heigth_to_resize):
        img = cv2.imread(img_path)
        assert width==img.shape[1]
        assert height==img.shape[0]
        #resized_width, resized_height=self.get_new_img_size(width,height)
        img = cv2.resize(img, (width_to_resize, heigth_to_resize), interpolation=cv2.INTER_CUBIC)
        size =[heigth_to_resize, heigth_to_resize]
        img = img[:, :, (2, 1, 0)]  # BGR -> RGB
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 2, 3, 1))

        return img

    def get_new_img_size(self,width, height, img_min_side=600):
        if width <= height:
            f = float(img_min_side) / width
            resized_height = int(f * height)
            resized_width = img_min_side
        else:
            f = float(img_min_side) / height
            resized_width = int(f * width)
            resized_height = img_min_side

        return resized_width, resized_height

    def get_img_output_length_res50(self,width, height):
        def get_output_length(input_length):
            # zero_pad
            input_length += 6
            # apply 4 strided convolutions
            filter_sizes = [7, 3, 1, 1]
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size + stride) // stride
            return input_length

        return get_output_length(width), get_output_length(height)

    def get_img_output_length_vgg(self,width, height):
        def get_output_length(input_length):
            return input_length // 16

        return get_output_length(width), get_output_length(height)







        #boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间
#label的内同是一个数
def match(boxlist,label):
    labellist= []
    boxdict ={}
    labelnp = np.zeros(class_num + 1)
    for i in range(7):
        labellist.append(labelnp)

    if len(labellist)!=7:
        raise ValueError('SDFA')
    for box in boxlist:
        index=part_map_num[box['name']]
        labellist[index][0]=1
        labellist[index][label]=1
        x = box['cor'][0]
        y = box['cor'][1]
        w = box['cor'][2]
        h = box['cor'][3]
        x *= net_size[1]
        w *=net_size[1]
        y *= net_size[0]
        h *=net_size[0]
        cor_np  = np.array([x,y,w,h])
        cor_np =np.expand_dims(cor_np, axis=0)
        cor_np = np.expand_dims(cor_np, axis=0)
        boxdict[box['name']] = cor_np



    return boxdict,labellist