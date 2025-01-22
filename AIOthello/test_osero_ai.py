# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:38:25 2021

@author: 84840
"""
import tensorflow as tf
import numpy as np
import random
import time


from osero_board import State, random_action, BOARD_LENGTH
model = tf.keras.models.load_model('osero_model_6.h5')
INPUT_SIZE = (BOARD_LENGTH, BOARD_LENGTH, 2)
BATTLE_COUNT = 100
ahodo = 2


#勝敗判定
def judge_winner(state):
    if state.is_first_player():
        if state.is_lose():
            return -1
        if state.is_draw():
            return 0
        else:
            return 1
    else:
        if state.is_lose():
            return 1
        if state.is_draw():
            return 0
        else:
            return -1

#ラベル付け
def add_learn(log,learn):
    new_log = []
    for i in log:
        i.append(learn)
        new_log.append(i)
    return new_log

#手の選択（１手進めてその局面が相手にとって悪くなる手を選択）
def make_action(state):
  state_copy = State(state.pieces.copy,state.enemy_pieces.copy,state.depth)
  leagal_board = []
  for leagal in state.legal_actions():
    leagal_board.append([(state.next(leagal)).pieces,(state.next(leagal)).enemy_pieces])
  leagal_board = np.array(leagal_board)
  leagal_board = leagal_board.reshape(len(leagal_board), INPUT_SIZE[2], INPUT_SIZE[0], INPUT_SIZE[1]).transpose(0, 2, 3, 1)
  score = model.predict(leagal_board)
  score = score.tolist()
  action = state.legal_actions()[score.index(min(score))]
  return action

#データ生成
def make_datas():
    log=[]
    timer_aa = 0#タイム計測
    timer_ab = 0
    timer_ac = 0
    timer_ad = 0
    timer_ada = 0
    timer_adb = 0
    timer_adc = 0
    timer_add = 0
    timer_ae = 0
    timer_af = 0
    timer_ag = 0
    timer_s = 0#タイム計測
    if True:
        states = [State() for _ in range(BATTLE_COUNT)]
        firster_log = [[] for _ in range(BATTLE_COUNT)]
        seconder_log = [[] for _ in range(BATTLE_COUNT)]
        end = 0
        while end != BATTLE_COUNT:
          
          start = time.time()
          legal_lists = []
          legal_len = [0] * BATTLE_COUNT
          end_time = time.time() - start
          timer_aa += end_time

          for i,state in enumerate(states):
            if state == None:
              continue
            if state.is_done():

                start = time.time()
                if judge_winner(state) == 1:
                    log += add_learn(firster_log[i],1)
                    log += add_learn(seconder_log[i],-1)
                if judge_winner(state) == 0:
                    log += add_learn(firster_log[i]+seconder_log[i],0)
                if judge_winner(state) == -1:
                    log += add_learn(seconder_log[i],1)
                    log += add_learn(firster_log[i],-1)
                states[i] = None
                end += 1
                end_time = time.time()
                timer_ab += end_time - start

                continue
            
            start = time.time()
            if state.is_first_player():
              firster_log[i].append([state.pieces,state.enemy_pieces])
            else:
              seconder_log[i].append([state.pieces,state.enemy_pieces])
            end_time = time.time() - start
            timer_ac += end_time
            
            start = time.time()
            if random.random()*10 > ahodo:
              legal_list = state.legal_actions()
              end_time = time.time()
              timer_ada += time.time() - start
              legal_len[i] = len(legal_list)
              end_time = time.time()
              timer_adb += time.time() - start
              for i in legal_list:
                legal_lists.append([(state.next(i)).pieces, (state.next(i)).enemy_pieces])
            end_time = time.time() - start
            timer_ad += end_time
            
          if legal_lists != []:
            start = time.time()
            legal_lists = tf.constant(legal_lists)
            legal_lists = tf.reshape(legal_lists,(len(legal_lists), INPUT_SIZE[2], INPUT_SIZE[0], INPUT_SIZE[1]))
            legal_lists = tf.transpose(legal_lists,perm=[0, 2, 3, 1])
            end_time = time.time() - start
            timer_ae += end_time


            start = time.time()#タイム計測
            scores = model.predict_on_batch(legal_lists)
            end_time = time.time() - start#タイム計測
            timer_s += end_time

            start = time.time()
            scores.flatten()
            scores = scores.tolist()
            end_time = time.time() - start
            timer_af += end_time

          start = time.time()
          score_list = [[] for _ in range(BATTLE_COUNT)]
          for i,j in enumerate(legal_len):
            if states[i] == None:
              continue
            for k in range(j):
              score_list[i].append(scores[k])
            del scores[:j]
            if score_list[i] != []:
              action_index = score_list[i].index(min(score_list[i]))
              action = states[i].legal_actions()[action_index]
            else:
              action = random_action(states[i])
            states[i] = states[i].next(action)
          end_time = time.time() - start
          timer_ag += end_time

    print(f"suiron{timer_s}")#タイム計測
    print(f"aa{timer_aa}")#タイム計測
    print(f"ab{timer_ab}")#タイム計測
    print(f"ac{timer_ac}")#タイム計測
    print(f"ad{timer_ad}")#タイム計測
    print(f"ada{timer_ada}")#タイム計測
    print(f"adb{timer_adb}")#タイム計測
    print(f"ae{timer_ae}")#タイム計測
    print(f"af{timer_af}")#タイム計測
    print(f"ag{timer_ag}")#タイム計測
    return log


DN_INPUT_SHAPE = (8,8,2)
import numpy as np
def alpha_beta(state, model, alpha, beta, reed_depth ,depth_limit=2):
    if state.is_done():

        if state.is_lose():
            return -1
        if state.is_draw():
            return 0
        return 1
    reed_depth += 1
    if reed_depth == depth_limit:
        a, b, c = DN_INPUT_SHAPE
        x = np.array([state.pieces, state.enemy_pieces])
        x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)
        return model.predict(x, batch_size=1)[0][0]

    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), model, -beta, -alpha, reed_depth)
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return  alpha
    return alpha
    
    

def alpha_beta_action(state,model):
    reed_depth = 0
    best_action = 0
    alpha = -float("inf")
    str = ["",""]
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), model, -float("inf"), -alpha, reed_depth)
        if score > alpha:
            best_action = action
            alpha =score
            
    return best_action


def test():
    log=[]
    firster_log = []
    seconder_log = []
    for _ in range(BATTLE_COUNT):
        state = State()
        while True:
            if state.is_done():
                print(judge_winner(state))
                if judge_winner(state) == 1:
                    log.append(10000)
                if judge_winner(state) == 0:
                    log.append(1)
                if judge_winner(state) == -1:
                    log.append(100)
                break
            #q=random.random()*10
            if state.is_first_player():#q>ahodo and 
                start = time.time()
                action = alpha_beta_action(state, model)
                print("time", time.time()-start)
            else:
                action = random_action(state)
            state = state.next(action)
    return log
log = test()
point = 0
for i in log:
  point += i
print(point)
#print(0.5+(point/len(log))/2)