from tensorflow.keras.models import load_model
from osero_board import State,random_action
from pv_mcts import pv_mcts_action
#from pv_mcts_100 import pv_mcts_action2
def test(a):
    model = load_model("model_best_3.h5")
    model2 = load_model("model_best_4.h5")
    
    next_action = pv_mcts_action(model, 1.0)
    next_action2 = pv_mcts_action(model2, 1.0)
    n = 0
    for i in range(1):
        state =State()                
        while True:
            if state.is_done():
                if state.is_lose():
                    if not state.is_first_player():
                        n += 1
                elif state.is_draw():
                    n += 0.5
                else:
                    if state.is_first_player():
                        n += 1                    
                break
            if state.is_first_player():
                action = next_action(state)
            else:
                action = next_action2(state)
                #action = random_action(state)            
            state = state.next(action)
            #print(state)
    return n

if __name__ =="__main__":
    N = 20#試行回数
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    import tensorflow as tf
    from concurrent import futures
    from tensorflow.keras import backend as K
    with tf.device('/CPU:0'):
        with futures.ProcessPoolExecutor(max_workers=5) as executor:
            h = executor.map(test, range(N), chunksize=1)
    historys = []
    for i in h:
        historys.append(i)

    #K.clear_session()

    print(sum(historys)/N)