# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np


result_class_path = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part_result{}.hdf5'.format(18)
with open(result_class_path,'r') as f:
    result_list = pickle.load(f)
#threshold = 0.1
for nidx in range(len(result_list)):
    netoutput = result_list[nidx]['netout']
    labellist = result_list[nidx]['label']
    for i in range(7):
        net_predict = netoutput[i][0,:]
        label = labellist[i][0,1:]
        pre_idx = np.argmax(net_predict)
        lab_idx = np.argmax(label)
        #one_sample = {'pre':net_predict,'label':labellist}
        if labellist[i][0, 0] == 1:
            if pre_idx ==lab_idx:
                tru_s[i]+=1
            else:
                fal_s[i]+=1
tru_s=np.zeros([7],dtype=np.float32)
fal_s=np.zeros([7],dtype=np.float32)
for nn in range(len(result_list)):

    for i in range(7):
        net_predict = result_list[nn]['pre']
        labellist = result_list[nn]['label']
        label = labellist[i][0,1:]
        pre_idx = np.argmax(net_predict)
        lab_idx = np.argmax(label)
        #one_sample = {'pre':net_predict,'label':labellist}
        #holy_pre_result.append(one_sample)
        if labellist[i][0, 0] == 1:
            if pre_idx ==lab_idx:
                tru_s[i]+=1
            else:
                fal_s[i]+=1
arc = np.zeros([7],dtype=np.float32)
for j in range(7):
    arc[j]= float(tru_s[j])/float(tru_s[j]+fal_s[j])
print(arc)