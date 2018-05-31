# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np


result_class_path = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part_result{}.hdf5'.format(16)
with open(result_class_path,'r') as f:
    result_list = pickle.load(f)
threshold = 0.1
