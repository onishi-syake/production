# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 01:01:59 2021

@author: 84840
"""

from tensorflow.keras.layers import Activation,Dense,Input,Flatten,Dropout,Conv2D,MaxPool2D,BatchNormalization,Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1,l2
import tensorflow as tf
import numpy as np
import random
import time

BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH ** 2

"""import tensorflow as tf
for _ in range(1):#後で消す
  start = time.time()
  input = Input(shape=[BOARD_LENGTH,BOARD_LENGTH,2])
  x = input
  for i in (32,64,128):
    x = Conv2D(i, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(i, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.25)(x)
  x = Flatten()(input)
  x = Dense(16)(x)
  x = Activation("relu")(x)
  x = Dense(1)(x)
  model = Model(inputs=input,outputs=x)
  model.compile(optimizer=Adam(),loss="mse",metrics="mae")"""
  
model = tf.keras.models.load_model('osero_model_6.h5')
  
import tensorflow as tf
import time
start = time.time()
f = tf.TensorArray(tf.bool, size=BOARD_SIZE)
for q in range(BOARD_SIZE):
  f = f.write(q, False)
dxy = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, -1], [0, -1], [1, -1], [-1, 0]]
a = tf.TensorArray(tf.bool, size=8*BOARD_LENGTH**2*(BOARD_LENGTH-1))
for i in range(BOARD_LENGTH-1):
  for j, xy in enumerate(dxy):
    for y in range(BOARD_LENGTH):#y
      for x in range(BOARD_LENGTH):#x
        if y + xy[1] * (i+2) < 0 or y + xy[1] * (i+2) > BOARD_LENGTH:
          a = a.write(i*BOARD_SIZE*8+j*BOARD_SIZE+y*BOARD_LENGTH+x, tf.fill([BOARD_LENGTH, BOARD_LENGTH], False))
          continue
        if x + xy[0] * (i+2) < 0 or x + xy[0] * (i+2) > BOARD_LENGTH:
          a = a.write(i*BOARD_SIZE*8+j*BOARD_SIZE+y*BOARD_LENGTH+x, tf.fill([BOARD_LENGTH, BOARD_LENGTH], False))
          continue
        b = tf.TensorArray(tf.bool, size=BOARD_SIZE)
        for q in range(BOARD_SIZE):
          b = b.write(q, False)
        for k in range(i+2):
          b = b.write((x+xy[0]*k)+(y+xy[1]*k)*BOARD_LENGTH, True)
        b = b.stack()
        b = tf.reshape(b, [BOARD_LENGTH, BOARD_LENGTH])
        a = a.write(i*BOARD_SIZE*8+j*BOARD_SIZE+y*BOARD_LENGTH+x, b)
a = a.stack()
FLIP_BOARD = tf.reshape(a, [BOARD_LENGTH-1, 8, BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH])
FLIP_BOARD = tf.transpose(FLIP_BOARD, [0, 4, 5, 1, 2, 3])#深さ、盤、方向、位置
F_FLIP_BOARD = tf.fill([BOARD_LENGTH, BOARD_LENGTH], False)
pass_board_t = tf.constant([True])
pass_board_f = tf.constant([False]*(BOARD_SIZE-1))
pass_board = tf.concat([pass_board_t, pass_board_f], 0)
pass_board = tf.reshape(pass_board, [BOARD_LENGTH, BOARD_LENGTH, 1])

import tensorflow as tfFLIP_BOARD
import numpy as np
import time
PLAY_GAMES = 200
class State:
    def __init__(self, pieces=None, depth=0, end_turn=None):
      # 方向定数
      self.dxy = tf.constant([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, -1], [0, -1], [1, -1], [-1, 0]])
      self.dxy = tf.cast(self.dxy, tf.int64)
      self.ddxy = tf.constant([[[i, 0], [i, i], [0, i], [-i, i], [-i, -i], [0, -i], [i, -i], [-i, 0]] for i in range(1,BOARD_LENGTH)])
      #False_Tensor_Array
      self.False_Tensor_Array = tf.TensorArray(tf.bool, size=BOARD_SIZE, clear_after_read=False)
      for i in range(BOARD_SIZE):
        self.False_Tensor_Array = self.False_Tensor_Array.write(i, False)
      #連続パスによる終了
      self.Pass = tf.constant([False] * PLAY_GAMES)
      self.pass_end = tf.constant([False] * PLAY_GAMES)

      #石の配置
      self.pieces = pieces
      self.depth = depth
      self.end_turn = end_turn

      #石の初期配置
      if pieces == None:
        self.pieces = [False] * BOARD_SIZE * 2
        self.pieces[int(BOARD_SIZE / 2 - BOARD_LENGTH / 2 - 1)] = self.pieces[int(BOARD_SIZE / 2 + BOARD_LENGTH / 2)] = True
        self.pieces[int(BOARD_SIZE * 3 / 2 - BOARD_LENGTH / 2)] = self.pieces[int(BOARD_SIZE * 3 / 2 + BOARD_LENGTH / 2 - 1)] = True
        self.pieces = tf.constant([self.pieces for _ in range(PLAY_GAMES)])
        self.end_turn = tf.constant([0] * PLAY_GAMES)
        self.kihu = None
        self.keep_kihu = None

    #勝敗判定(先手にとっての)
    def judge(self,pieces):
      pieces = tf.cast(pieces, tf.float32)
      count = tf.math.reduce_sum(pieces[:, 0:BOARD_SIZE], axis=1)
      enemy_count = tf.math.reduce_sum(pieces[:, BOARD_SIZE:BOARD_SIZE*2], axis=1)
      jud = count - enemy_count
      w_jud = jud > 0
      w_jud = tf.cast(w_jud, tf.float32)
      l_jud = jud < 0
      l_jud = tf.cast(l_jud, tf.float32)
      jud = w_jud - l_jud
      if not self.is_first_player():
        jud *= -1
      return jud
      
    #個別終了判定（出力はTrueかFalseが盤の数分だけ出力）
    def is_done(self):
      pieces = tf.cast(self.pieces, tf.float32)
      all_filled_end = tf.math.reduce_sum(pieces, axis=1) == float(BOARD_SIZE)
      all_filled_end = tf.cast(all_filled_end, tf.float32)
      pass_end = tf.cast(self.pass_end, tf.float32)
      end = all_filled_end + pass_end
      end = end > 0
      return end

    #終了棋譜保存/削除
    def drop(self):
      if self.pieces.shape == tf.constant([]).shape:
          pass
      else:
          end_bool_list = self.is_done()
          end_list = tf.where(end_bool_list)[:,0]
          #保存
          end_kihu = self.kihu[end_bool_list]#先後ラベル付き(3(2)次元)
          end_rabel = end_kihu[:,:,BOARD_SIZE*2]#先後ラベル(2(1)次元)
          jud_rabel = self.judge(self.pieces[end_bool_list])#勝敗ラベル（1（0）次元）
          if jud_rabel.shape == tf.constant([]).shape:
              pass
          else:
              jud_rabel = tf.reshape(jud_rabel, (len(jud_rabel), 1))#2(1)次元化
              jud_rabel = end_rabel * jud_rabel#先後ラベルと勝敗ラベルを掛け合わせる
              jud_rabel = tf.reshape(jud_rabel,(len(jud_rabel), len(jud_rabel[0]), 1))#3(2)次元化
              end_kihu = tf.concat([end_kihu[:,:,:BOARD_SIZE*2],jud_rabel],2)#先後ラベルから勝敗ラベルに張り替え
              end_kihu = tf.reshape(end_kihu,(len(end_kihu)*len(end_kihu[0]),BOARD_SIZE*2+1))#2(1)次元化
              if type(self.keep_kihu) == type(None):
                self.keep_kihu = end_kihu
              else:
                self.keep_kihu = tf.concat([self.keep_kihu, end_kihu], 0)
              #削除
              self.pieces = np.delete(self.pieces, end_list, axis=0)
              self.kihu = np.delete(self.kihu, end_list, axis=0)
              self.pass_end = np.delete(self.pass_end, end_list, axis=0)
              self.Pass = np.delete(self.Pass, end_list, axis=0)

    #先手かどうか（TrueかFalse）
    def is_first_player(self):
      return self.depth % 2 == 0

    #棋譜の保存(先後ラベル付き/棋譜はfloat型)
    def keep(self):
      if self.pieces.shape == tf.constant([]).shape:
          pass
      else:
          rabel = [[float(self.is_first_player()) * 2 - 1]] * self.pieces.shape[0]
          rabel = tf.constant(rabel)
          pieces = tf.cast(self.pieces, tf.float32)
          pieces_r = tf.concat([pieces,rabel],1)
          if type(self.kihu) == type(None):
            self.kihu = tf.reshape(pieces_r, (len(pieces_r), 1, BOARD_SIZE*2+1))
          else:
            pieces_r = tf.reshape(pieces_r, (len(pieces_r),1,BOARD_SIZE*2+1))
            self.kihu = tf.concat([self.kihu, pieces_r], 1)

    #合法手取得(TrueかFalseを盤と同じ形で出力)
    def legal_actions(self):
      pieces = tf.reshape(self.pieces, [self.pieces.shape[0], 2, BOARD_LENGTH, BOARD_LENGTH])
      pieces = tf.transpose(pieces, perm=[0, 2, 3, 1])
      filled_pieces = tf.math.logical_or(pieces[:,:,:,0], pieces[:,:,:,1])
      empty_pieces = tf.math.logical_not(filled_pieces)
      p_pieces = tf.cast(pieces, tf.float32)
      p_pieces = tf.keras.layers.ZeroPadding2D(padding=BOARD_LENGTH-1)(p_pieces)
      p_pieces = tf.cast(p_pieces, tf.bool)
      self.p_pieces = p_pieces
      t_pieces = tf.map_fn(self.cut_move_1, self.ddxy, parallel_iterations=8*(BOARD_LENGTH-1), fn_output_signature=tf.bool)
      
      #合法手探索
      empty_8 = tf.tile(empty_pieces, tf.constant([8,1,1]))
      empty_8 = tf.reshape(empty_8, [8, empty_pieces.shape[0], BOARD_LENGTH, BOARD_LENGTH])
      searching = tf.math.logical_and(empty_8, t_pieces[0,:,:,:,:,1])
      can_flip_1 = tf.math.logical_and(searching, t_pieces[1,:,:,:,:,0])#方、数、位
      B_FLIP_BOARD = tf.expand_dims(FLIP_BOARD, 0)
      pri = tf.transpose(B_FLIP_BOARD, [0, 1, 2, 5, 6, 3, 4])
      B_FLIP_BOARD = tf.tile(B_FLIP_BOARD, tf.constant([pieces.shape[0], 1, 1, 1, 1, 1, 1]))#xxx数、深、方、盤、位, ooo数、深、盤、方、位
      B_FLIP_BOARD = tf.transpose(B_FLIP_BOARD, [1, 2, 3, 4, 0, 5, 6])#xxx深、方、盤、数、位,ooo深、盤、方、数、位
      flip_point = tf.where(can_flip_1, B_FLIP_BOARD[0], F_FLIP_BOARD)#盤、方、数、位
      flip_points = tf.expand_dims(flip_point, 0)#深、盤、方、数、位
      can_flip_2 = can_flip_1 #方、数、位
      for i in tf.range(BOARD_LENGTH-3):
        searching = tf.math.logical_and(searching, t_pieces[1+i,:,:,:,:,1])
        can_flip_1 = tf.math.logical_and(searching, t_pieces[2+i,:,:,:,:,0])
        flip_point = tf.where(can_flip_1, B_FLIP_BOARD[i], F_FLIP_BOARD)#盤、方、数、位
        flip_point = tf.expand_dims(flip_point, 0)#深、盤、方、数、位
        flip_points = tf.concat([flip_points, flip_point], 0)#深、盤、方、数、位
        can_flip_2 = tf.math.logical_or(can_flip_2, can_flip_1)
      flip_points = tf.transpose(flip_points, [0, 3, 4, 5, 6, 1, 2])#深、方、数、位、盤
      flip_points = tf.reduce_any(flip_points, 0)#方、数、位、盤
      flip_points = tf.reduce_any(flip_points, 0)#数、位、盤
      judger = tf.reduce_any(can_flip_2, 0)#数、位
      judger = tf.reshape(judger, [judger.shape[0], BOARD_LENGTH, BOARD_LENGTH])
      return judger, flip_points#盤数、位置、盤という順番

    #３手読み用（6，6，2）の盤を入れる
    def tri_legal_actions(self, pieces):
      filled_pieces = tf.math.logical_or(pieces[:,:,:,0], pieces[:,:,:,1])
      empty_pieces = tf.math.logical_not(filled_pieces)
      p_pieces = tf.cast(pieces, tf.float32)
      p_pieces = tf.keras.layers.ZeroPadding2D(padding=BOARD_LENGTH-1)(p_pieces)
      p_pieces = tf.cast(p_pieces, tf.bool)
      self.p_pieces = p_pieces
      t_pieces = tf.map_fn(self.cut_move_1, self.ddxy, parallel_iterations=8*(BOARD_LENGTH-1), fn_output_signature=tf.bool)
      
      #合法手探索
      empty_8 = tf.tile(empty_pieces, tf.constant([8,1,1]))
      empty_8 = tf.reshape(empty_8, [8, empty_pieces.shape[0], BOARD_LENGTH, BOARD_LENGTH])
      searching = tf.math.logical_and(empty_8, t_pieces[0,:,:,:,:,1])
      can_flip_1 = tf.math.logical_and(searching, t_pieces[1,:,:,:,:,0])#方、数、位
      B_FLIP_BOARD = tf.expand_dims(FLIP_BOARD, 0)
      pri = tf.transpose(B_FLIP_BOARD, [0, 1, 2, 5, 6, 3, 4])
      B_FLIP_BOARD = tf.tile(B_FLIP_BOARD, tf.constant([pieces.shape[0], 1, 1, 1, 1, 1, 1]))#xxx数、深、方、盤、位, ooo数、深、盤、方、位
      B_FLIP_BOARD = tf.transpose(B_FLIP_BOARD, [1, 2, 3, 4, 0, 5, 6])#xxx深、方、盤、数、位,ooo深、盤、方、数、位
      flip_point = tf.where(can_flip_1, B_FLIP_BOARD[0], F_FLIP_BOARD)#盤、方、数、位
      flip_points = tf.expand_dims(flip_point, 0)#深、盤、方、数、位
      can_flip_2 = can_flip_1 #方、数、位
      for i in tf.range(BOARD_LENGTH-3):
        searching = tf.math.logical_and(searching, t_pieces[1+i,:,:,:,:,1])
        can_flip_1 = tf.math.logical_and(searching, t_pieces[2+i,:,:,:,:,0])
        flip_point = tf.where(can_flip_1, B_FLIP_BOARD[i], F_FLIP_BOARD)#盤、方、数、位
        flip_point = tf.expand_dims(flip_point, 0)#深、盤、方、数、位
        flip_points = tf.concat([flip_points, flip_point], 0)#深、盤、方、数、位
        can_flip_2 = tf.math.logical_or(can_flip_2, can_flip_1)
      flip_points = tf.transpose(flip_points, [0, 3, 4, 5, 6, 1, 2])#深、方、数、位、盤
      flip_points = tf.reduce_any(flip_points, 0)#方、数、位、盤
      flip_points = tf.reduce_any(flip_points, 0)#数、位、盤
      judger = tf.reduce_any(can_flip_2, 0)#数、位
      judger = tf.reshape(judger, [judger.shape[0], BOARD_LENGTH, BOARD_LENGTH])
      return judger, flip_points#盤数、位置、盤という順番

    #切り出して動かす
    #@tf.function
    def cut_move_1(self,ddxy):#p_piecesはグローバルから参照
      return tf.map_fn(self.cut_move_2, ddxy, parallel_iterations=8, fn_output_signature=tf.bool)

    def cut_move_2(self, dxy):
      return self.p_pieces[:, BOARD_LENGTH-1+dxy[1]:BOARD_LENGTH*2+dxy[1]-1, BOARD_LENGTH-1+dxy[0]:BOARD_LENGTH*2+dxy[0]-1, :]



    #n_nextの改良版（合法手なしならばパス）
    def p_next(self):
      judger_bool, flip_points = self.legal_actions()#数、位　　数、位、盤
      f_legal_list = tf.cast(judger_bool, tf.int32)
      f_legal_list = tf.reshape(f_legal_list, [judger_bool.shape[0], BOARD_SIZE])
      legal_board_num = tf.reduce_sum(f_legal_list, 1)
      legal_board_bool = tf.cast(legal_board_num, tf.bool)#1次元、shape=盤数（合法手の有無）
      tra_judger_bool = tf.transpose(judger_bool, [1, 2, 0])#位、数
      judger_bool = tf.where(legal_board_bool, tra_judger_bool, pass_board)#位、数
      judger_bool = tf.transpose(judger_bool, [2, 0, 1])#数、位
      n_legal_board_bool = tf.math.logical_not(legal_board_bool)
      n_legal_board_int = tf.cast(n_legal_board_bool, tf.int32)#合法手のない盤を1、他を0
      self.pass_end = tf.logical_and(self.Pass, n_legal_board_bool)
      self.Pass = n_legal_board_bool
      legal_board_num = legal_board_num + n_legal_board_int
      flip_points = tf.expand_dims(flip_points, axis=-1)
      pieces = tf.reshape(self.pieces, [self.pieces.shape[0], 2, BOARD_LENGTH, BOARD_LENGTH])
      pieces = tf.transpose(pieces, perm=[0, 2, 3, 1])
      pieces = tf.tile(pieces, tf.constant([1, BOARD_SIZE, 1, 1]))
      pieces = tf.reshape(pieces, [pieces.shape[0], BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH, 2, 1])
      m_pieces = pieces[:, :, :, :, :, 0]
      m_pieces = tf.math.logical_or(m_pieces, flip_points)
      e_flip_points = tf.math.logical_not(flip_points)
      e_pieces = pieces[:, :, :, :, :, 1]
      e_pieces = tf.math.logical_and(e_pieces, e_flip_points)
      legal_pieces = tf.concat([e_pieces, m_pieces], -1)
      #３手読み用
      legal_board_place = tf.where(judger_bool)
      legal_board_place = legal_board_place[:, 0]#shape=合法手の数    （合法手の元の盤）
      return legal_pieces[judger_bool], legal_board_num, legal_board_place

    #３手読み用(6,6,2)の盤を入力
    def tri_p_next(self, pieces):
      start = time.time()
      judger_bool, flip_points = self.tri_legal_actions(pieces)#数、位　　数、位、盤
      f_legal_list = tf.cast(judger_bool, tf.int32)
      f_legal_list = tf.reshape(f_legal_list, [judger_bool.shape[0], BOARD_SIZE])
      legal_board_num = tf.reduce_sum(f_legal_list, 1)
      legal_board_bool = tf.cast(legal_board_num, tf.bool)#1次元、shape=盤数（合法手の有無）
      tra_judger_bool = tf.transpose(judger_bool, [1, 2, 0])#位、数
      judger_bool = tf.where(legal_board_bool, tra_judger_bool, pass_board)#位、数
      judger_bool = tf.transpose(judger_bool, [2, 0, 1])#数、位
      n_legal_board_bool = tf.math.logical_not(legal_board_bool)
      n_legal_board_int = tf.cast(n_legal_board_bool, tf.int32)#合法手のない盤を1、他を0
      legal_board_num = legal_board_num + n_legal_board_int
      flip_points = tf.expand_dims(flip_points, axis=-1)
      pieces = tf.tile(pieces, tf.constant([1, BOARD_SIZE, 1, 1]))
      pieces = tf.reshape(pieces, [pieces.shape[0], BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH, BOARD_LENGTH, 2, 1])
      m_pieces = pieces[:, :, :, :, :, 0]
      m_pieces = tf.math.logical_or(m_pieces, flip_points)
      e_flip_points = tf.math.logical_not(flip_points)
      e_pieces = pieces[:, :, :, :, :, 1]
      e_pieces = tf.math.logical_and(e_pieces, e_flip_points)
      legal_pieces = tf.concat([e_pieces, m_pieces], -1)
      #３手読み用
      legal_board_place = tf.where(judger_bool)
      legal_board_place = legal_board_place[:, 0]#shape=合法手の数    （合法手の元の盤）
      return legal_pieces[judger_bool], legal_board_num, legal_board_place


      
    #３手読み用
    def tri_predict(self):
      legal_pieces_1, legal_board_num_1, _ = self.p_next()
      legal_pieces_2, _, legal_board_place_2 = self.tri_p_next(legal_pieces_1)
      legal_pieces_3, _, legal_board_place_3 = self.tri_p_next(legal_pieces_2)
      score_3 = model(legal_pieces_3, training=False)
      score_2 = tf.math.segment_min(score_3, legal_board_place_3, name=None)
      score_1 = tf.math.segment_max(score_2, legal_board_place_2, name=None)
      scores = tf.reshape(score_1, [score_1.shape[0]])
      print("legal_board_num_1", legal_board_num_1)
      next = tf.TensorArray(tf.bool, size=legal_board_num_1.shape[0])
      for i,j in enumerate(legal_board_num_1):
          if random.random() > 0.95**self.depth:
              next = next.write(i, legal_pieces_1[tf.math.argmin(scores[:j])])
          else:
              next = next.write(i, legal_pieces_1[random.randint(0, j-1)])
          legal_pieces_1 = legal_pieces_1[j-1:]
          scores = scores[j-1:]
      next = next.stack()
      if next.shape == tf.constant([]).shape:
          return next
      next = tf.transpose(next, perm=[0, 3, 1, 2])
      return tf.reshape(next, [next.shape[0], BOARD_SIZE*2])




    def predicter(self, down_greedy):
      next_list, legal_board_num, _ = self.p_next()
      #scores = self.predict(next_list)
      scores = model(next_list, training=False)
      scores = tf.reshape(scores, [next_list.shape[0]])
      next = tf.TensorArray(tf.bool, size=legal_board_num.shape[0])
      for i,j in enumerate(legal_board_num):
          if random.random() > (0.95**self.depth)*down_greedy:
              next = next.write(i, next_list[tf.math.argmin(scores[:j])])
          else:
              next = next.write(i, next_list[random.randint(0, j-1)])
          next_list = next_list[j-1:]
          scores = scores[j-1:]
      next = next.stack()
      if next.shape == tf.constant([]).shape:
          return next
      next = tf.transpose(next, perm=[0, 3, 1, 2])
      return tf.reshape(next, [next.shape[0], BOARD_SIZE*2])

      #return tf.constant(next)

    #@tf.function
    def predict(self, next_list):
      return model(next_list, training=False)