# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:37:58 2021

@author: 84840
"""

import tensorflow as tf
from multiprocessing import Pool
import os
import time

BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH ** 2

def arg(datas):
    print(datas)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    with Pool(8) as p:
        flat_datas = p.map(flat, datas)#x, BOARD_SIZE*3+1
    flat_datas = tf.constant(flat_datas)
    print(flat_datas.shape)
    flat_datas = tf.reshape(flat_datas, [flat_datas.shape[0], BOARD_SIZE*3+1])
    datas_arg = flat_datas[:, :BOARD_SIZE*3]
    datas_rabel = flat_datas[:, BOARD_SIZE*3]
    datas_arg = tf.reshape(datas_arg, [flat_datas.shape[0], 3, BOARD_LENGTH, BOARD_LENGTH])
    datas_arg = tf.transpose(datas_arg, [0, 2, 3, 1])
    datas_arg = perfect_gen(datas_arg)
    datas_arg = tf.transpose(datas_arg, [0, 3, 1, 2])
    datas_arg = tf.reshape(datas_arg, [datas_arg.shape[0], BOARD_SIZE*3])
    datas_rabel = tf.tile(datas_rabel, [8])
    datas_rabel = tf.expand_dims(datas_rabel, 1)
    print(datas_arg.shape, datas_rabel.shape)
    arg_datas = tf.concat([datas_arg, datas_rabel], axis=1)
    arg_datas = arg_datas.numpy().tolist()
    with Pool(8) as p:
        unflat_datas = p.map(unflat, arg_datas)
    return unflat_datas
    
def unflat(data):
    x = [[[], []], [], None]
    x[0][0].extend(data[:BOARD_SIZE])
    x[0][1].extend(data[BOARD_SIZE:BOARD_SIZE*2])
    x[1].extend(data[BOARD_SIZE*2:BOARD_SIZE*3])
    x[2] = data[BOARD_SIZE*3]
    return x
    
    
def flat(data):
    x = []
    x.extend([float(i) for i in data[0][0]])
    x.extend([float(j) for j in data[0][1]])
    x.extend(data[1])
    x.append(data[2])
    return x

def perfect_gen(train_log):
  l1 = tf.image.flip_left_right(train_log)
  l2 = tf.image.flip_up_down(train_log)
  l3 = tf.image.flip_left_right(l2)
  l4 = tf.image.rot90(train_log)
  l5 = tf.image.flip_left_right(l4)
  l6 = tf.image.flip_up_down(l4)
  l7 = tf.image.flip_left_right(l6)
  return tf.concat([train_log,l1,l2,l3,l4,l5,l6,l7], 0)

if __name__  == "__main__":
    #print(flat([[[1, 2], [3, 4]], [5], 6]))
    #print(unflat([1]*64+[2]*64+[3]*64+[4]))
    start = time.time()
    print(arg([unflat([1]*64+[2]*64+[3]*64+[4])]*10000))
    print(time.time()-start)