from osero_board import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from pathlib import Path
from concurrent import futures
import numpy as np
import pickle
import os

SP_GAME_COUNT = 100
SP_TEMPERATURE = 1.0

def first_player_value(ended_state):
  if ended_state.is_lose():
    return -1 if ended_state.is_first_player() else 1
  return 0

def write_data(history):
  now = datetime.now()
  os.makedirs("./data", exist_ok=True)
  path = ".data/{:04}{:02}{:02}{:02}{:02}{:02}.history".format(
      now.year, now.month, now.day, now.hour, now.minute, now.second)
  with open(path, mode='wb') as f:
      pickle.dump(history, f)
  
def play(_):
    history = []
    model = load_model("model_best_3.h5")
    state = State()
    while True:
        
        if state.is_done():
            break
        
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)
        
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])
        
        action = np.random.choice(state.legal_actions(), p=scores)
        
        state = state.next(action)
        
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    del model
    return history

def self_play():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    with tf.device('/CPU:0'):
        with futures.ProcessPoolExecutor(max_workers=5) as executor:
            
                h = executor.map(play, range(SP_GAME_COUNT), chunksize=1)
        historys = []
        for i in h:
            historys.extend(i)
    
        K.clear_session()
    
    
    return historys
    
if __name__ == "__main__":
    print(self_play())