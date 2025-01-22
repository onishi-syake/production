DN_INPUT_SHAPE = (8,8,2)
import numpy as np
def alpha_beta(state, model, alpha, beta, reed_depth ,depth_limit=3):
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
    alpha = float("inf")
    str = ["",""]
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), model, -float("inf"), -alpha, reed_depth)
        if score > alpha:
            best_action = action
            alpha =score
            
    return best_action