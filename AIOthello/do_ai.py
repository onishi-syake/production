# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:29:29 2021

@author: 84840
"""

from train_network import train_network
from self_play import self_play

if __name__ == "__main__":
    for i in range(50):
        print(i)
        data = self_play()
        print(i)
        train_network(data)
        print(i)