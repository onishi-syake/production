# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:55:29 2021

@author: 84840
"""

from train_network import train_network
from self_play import self_play
import tensorflow as tf
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)
if __name__ == "__main__":
    import time
    t = time.time()
    for i in range(50):
        print(i)
        data = self_play()
        train_network(data)
    print(time.time()-t)
    print("OWARI")