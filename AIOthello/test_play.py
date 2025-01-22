from tensorflow.keras.models import load_model
from osero_board import State,random_action
from pv_mcts import pv_mcts_action
if __name__ =="__main__":
    model_1 = load_model("model_best_3.h5")
    model_2 = load_model("model_best_t.h5")
    next_action_1 = pv_mcts_action(model_1, 1.0)
    next_action_2 = pv_mcts_action(model_2, 1.0)
    n = 0
    for i in range(10):
        state =State()
        while True:
            if state.is_done():
                print("e")
                if state.is_lose():
                    if not state.is_first_player():
                        print("w")
                        n += 1
                elif state.is_draw():
                    n += 0.5
                    print("d")
                else:
                    if state.is_first_player():
                        n += 1                    
                        print("w")
                break
            if state.is_first_player():
                action = next_action_1(state)
            else:
                action = next_action_2(state)            
            state = state.next(action)
            #print(state)
    print(n)