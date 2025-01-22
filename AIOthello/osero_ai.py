# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:03:00 2021

@author: 84840
"""

DN_INPUT_SHAPE = (8,8,2)
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
from osero_board import State
import numpy as np

PV_EVALUATE_COUNT = 50

def predict(model, state):
    
    a, b, c = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)
    y = model.predict(x, batch_size=1)
    
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1
    
    value = y[1][0][0]
    return policies, value

def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

def pv_mcts_scores(model, state, temperature):
    
    class node:
        def __init__(self, state, p):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = None
        
        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0
                self.w += value
                self.n += 1
                return value
            
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                self.w += value
                self.n += 1
                
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(node(self.state.next(action), policy))
                return value
            
            else:
                value = -self.next_child_node().evaluate()
                
                self.w += value
                self.n += 1
                return value
            
        def next_child_node(self):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) + 
                                   C_PUCT * child_node.p *sqrt(t) / (1 + child_node.n))
                
            return self.child_nodes[np.argmax(pucb_values)]
        
    root_node = node(state, 0)
    
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()
        
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores

def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

DN_FILTERS = 64
DN_RESIDUAL_NUM = 10
#DN_INPUT_SHAPE = (8,8,2)
DN_OUTPUT_SIZE = 37

def conv(filters):
    return Conv2D(filters, 3, padding="same", use_bias=False,
                  kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))

def residual_block():
  def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation("relu")(x)
        return x
  return f

def dual_network():
    if os.path.exists("./model/best.h5"):
        return
    
    input = Input(shape=DN_INPUT_SHAPE)
    
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    #p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              #activation="softmax", name="pi")(x)
    
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name="v")(v)
    
    model = Model(inputs=input, outputs=[v])#[p,v])
    
    #os.makedirs("./model/", exist_ok=True)
    #model.save("./model/best.h5")
    
    #K.clear_session()
    #del model
    return model

"""import time
if __name__ =="__main__":
    model = dual_network()
    state =State()
    
    next_action = pv_mcts_action(model, 1.0)    
    while True:
        t=time.time()

        if state.is_done():
            break
        action = next_action(state)
        state = state.next(action)
        print(state)
        #print(time.time()-t)"""

model = dual_network()
model.compile(optimizer=Adam(learning_rate=0.0001),loss="mse",metrics="mae")
model.save('osero_model_6.h5')