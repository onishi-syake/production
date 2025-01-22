# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 01:20:21 2021

@author: 84840
"""
from tensorflow.keras.models import load_model
from osero_board import State, random_action
from pv_mcts import pv_mcts_action
if __name__ =="__main__":
    model = load_model("model_best_t2.h5")
    state =State()
    
    next_action = pv_mcts_action(model, 1.0)
    import time
    t = time.time()
    while True:
        if state.is_done():
            break
        if state.is_first_player():
            action = next_action(state)
        else:
            action = random_action(state)            
        state = state.next(action)
        print(state)
    print(time.time() - t)