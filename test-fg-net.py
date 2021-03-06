# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

import cv2
import os
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn import march_ori as march



from keras_frcnn.pascal_voc_parser import get_data
def get_output_length(input_length):
    # zero_pad
    input_length += 6
    # apply 4 strided convolutions
    filter_sizes = [7, 3, 1, 1]
    stride = 2
    for filter_size in filter_sizes:
        input_length = (input_length - filter_size + stride) // stride
    return input_length
part_class_mapping={'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
bird_map_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
with open(bird_map_path,'r') as f:
    bird_class_mapping=pickle.load(f)
#pprint.pprint(bird_class_mapping)
def read_prepare_input(img_path):
    img = cv2.imread(img_path)
    return img
os.environ['CUDA_VISIBLE_DEVICES']= '1'

cfg = config.Config()
sys.setrecursionlimit(40000)


if cfg.network == 'vgg':
    # cfg.network = 'vgg'
    from keras_frcnn import vgg as nn
    print('use vgg')
elif cfg.network == 'resnet50':
    from keras_frcnn import resnet as nn
    print('use resnet50')
else:
    print('Not a valid model')
    raise ValueError

#cfg.base_net_weights = cfg.ori_res50_withtop

all_imgs, classes_count, bird_class_count = get_data(cfg.train_path,part_class_mapping)
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=False)
#pprint.pprint(classes_count)
#pprint.pprint(part_class_mapping)
# 这里的类在match里边定义
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    part_class_mapping['bg'] = len(part_class_mapping)

cfg.class_mapping = part_class_mapping


print('Training images per class:')
pprint.pprint(classes_count)
pprint.pprint(part_class_mapping)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print('Training bird per class:')
pprint.pprint(bird_class_count)
print('total birds class is {}'.format(len(bird_class_count)))
print('bird_class_mapping')
pprint.pprint(bird_class_mapping)


config_output_filename = cfg.config_filepath
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(cfg, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))
random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,K.image_dim_ordering(), mode='train')
#data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,K.image_dim_ordering(), mode='val')
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
#roi_input = Input(shape=(None, 4))  # roiinput是什么,要去看看清楚
part_roi_input = Input(shape=[None,4])

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)  # 共享网络层的输出.要明确输出的size
#bird_classifier_output = nn.fg_classifier(shared_layers,bird_rois_input0,bird_rois_input1,bird_rois_input2,bird_rois_input3,bird_rois_input4,bird_rois_input5,bird_rois_input6,nb_classes=200, trainable=True)
holyclass_out = nn.fine_layer(shared_layers, part_roi_input,nb_classes=200)

class_holyimg_out = nn.fine_layer_hole(shared_layers, part_roi_input,num_rois=1,nb_classes=200)

model_holyclassifier = Model([img_input,part_roi_input],holyclass_out)
#model_classifier_holyimg = Model([img_input,part_roi_input],class_holyimg_out)
cfg.base_net_weights = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part{}.hdf5'.format(18)
try:
    print('loading weights from {}'.format(cfg.base_net_weights))
    #model_rpn.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier.load_weights(cfg.base_net_weights, by_name=True)
    #model_birdclassifier.load_weights(cfg.base_net_weights, by_name=True)
    model_holyclassifier.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier_holyimg.load_weights(cfg.base_net_weights,by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
    raise ValueError('load wrong')
optimizer = Adam(lr=1e-5)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss())
model_holyclassifier.compile(optimizer=optimizer,loss=lossfn_list)
#model_classifier_holyimg.compile(optimizer=optimizer,loss=lossfn_list)

max_epoch=32
step= 0
now_epoch = 0
tru_s=np.zeros([7],dtype=np.int16)
fal_s=np.zeros([7],dtype=np.int16)
holy_pre_result = []
while 1:
    step+=1
    img_np,boxnp, labellist,img_path = data_lei.next_batch(1)
    #print(img_np.shape)
    #print(boxnp.shape)
    #input_img = read_prepare_input(img_path)
    #print(boxnp.shape)
    #print(img_path)
    #print(index)
    #exit(4)
    #print(labellist)
    #exit(4)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np,boxnp],labellist)
    netoutput = model_holyclassifier.predict_on_batch([img_np,boxnp])
    img = cv2.imread(img_path)
    img_ori = cv2.imread(img_path)
    size_w_ori = img_ori.shape[1]
    size_h_ori = img_ori.shape[0]
    size_w_map = get_output_length(img_np.shape[2])
    size_h_map = get_output_length(img_np.shape[1])
    for i in range(7):
        net_predict = netoutput[i][0,:]
        label = labellist[i][0,1:]
        pre_idx = np.argmax(net_predict)
        lab_idx = np.argmax(label)
        one_sample = {'pre':net_predict,'label':labellist}
        holy_pre_result.append(one_sample)
        if pre_idx ==lab_idx:
            tru_s[i]+=1
        else:
            fal_s[i]+=1
        for key, value in part_class_mapping.items():
            if value == i:
                pname = key
        if labellist[i][0, 0] == 1:
            baohan ='yes'
        else:
            baohan = 'no'
        print('part:{}  predict idx:{}  lab_idx:{} C :{} {}'.format(pname,pre_idx,lab_idx,net_predict[pre_idx],baohan))
        x1_out = boxnp[0, i, 0]
        y1_out = boxnp[0, i, 1]
        w_out = boxnp[0, i, 2]
        h_out = boxnp[0, i, 3]
        w = w_out * (size_w_ori / size_w_map)
        h = h_out * (size_h_ori / size_h_map)
        x1 = x1_out * (size_w_ori / size_w_map)
        y1 = y1_out * (size_h_ori / size_h_map)
        if labellist[i][0, 0] == 1:
            for key, value in part_class_mapping.items():
                if value == i:
                    pname = key
            # print('{}  {}  {}'.format(key, pre_idx, lab_idx))
            cv2.rectangle(img_ori, (int(x1), int(y1)), (int(x1 + w), int(y1 + w)), (0, 255, 0), 2)
            # cv2.rectangle(img_ori, (int(x1), int(y1) - 20),
            #            (int(x1 + w), int(y1)), (125, 125, 125), -1)
            cv2.putText(img_ori, '{} p:{} l:{} C:{}'.format(pname, pre_idx, lab_idx, net_predict[pre_idx]),
                        (int(x1) + 5, int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
    print('\n')
    '''for j in range(boxnp.shape[1]):
        x1_out = boxnp[0, j, 0]
        y1_out = boxnp[0, j, 1]
        w_out = boxnp[0, j, 2]
        h_out = boxnp[0, j, 3]
        w = w_out * (size_w_ori / size_w_map)
        h = h_out * (size_h_ori / size_h_map)
        x1 = x1_out * (size_w_ori / size_w_map)
        y1 = y1_out * (size_h_ori / size_h_map)
        if labellist[j][0, 0] == 1:
            for key, value in part_class_mapping.items():
                if value == j:
                    pname = key
            #print('{}  {}  {}'.format(key, pre_idx, lab_idx))
            cv2.rectangle(img_ori, (int(x1), int(y1)), (int(x1 + w), int(y1 + w)), (0, 255, 0), 2)
            #cv2.rectangle(img_ori, (int(x1), int(y1) - 20),
            #            (int(x1 + w), int(y1)), (125, 125, 125), -1)
            cv2.putText(img_ori, '{} p:{} l:{} C :{}'.format(pname, pre_idx, lab_idx, net_predict[pre_idx]), (int(x1) + 5, int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            print('part:{}  predict idx:{}  lab_idx:{} C :{}'.format(pname, pre_idx, lab_idx, net_predict[pre_idx]))'''

    # print()
    # continue
    # annot = img_annot
    # img_path = annot['filepath']
    #img_np = cv2.imread(img_path)
    # print(annot)
    # cv2.imshow('ori_label', img_np)
    cv2.imshow('net_label', img_ori)
    cv2.waitKey(0)
    #predict = model_holyclassifier.predict([img_np,boxnp])
    #predict =np.mean(predict,axis=1)
    #print(predict[0])
    #for i in range(7):
    #    print(np.argmax(predict[i],axis=1))
    #result = np.max(predict,axis=1)
    #print(predict.shape)
    #print(result)
    #exit()
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #A = model_holyclassifier.predict([img_np,boxnp])
    #print(holynet_loss)
    #exit()
    #print(boxnp)
    #print('step is {} batch_index is {} and epoch is {}'.format(step,data_lei.batch_index,data_lei.epoch))
    #print(holynet_loss)

    #break
    if data_lei.epoch!= now_epoch:
        arc = np.zeros([7],dtype=np.float32)
        for j in range(7):
            arc[j]= float(tru_s[j])/float(tru_s[j]+fal_s[j])
        print(arc)
        break
result_class_path = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part_result{}.hdf5'.format(18)
with open(result_class_path,'wb') as f:
    pickle.dump(holy_pre_result,f)

def get_output_length(input_length):
    # zero_pad
    input_length += 6
    # apply 4 strided convolutions
    filter_sizes = [7, 3, 1, 1]
    stride = 2
    for filter_size in filter_sizes:
        input_length = (input_length - filter_size + stride) // stride
    return input_length