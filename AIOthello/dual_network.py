from osero_board import State
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

DN_FILTERS = 64
DN_RESIDUAL_NUM = 10
DN_INPUT_SHAPE = (8,8,2)
DN_OUTPUT_SIZE = 65

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
    #if os.path.exists("model/best.h5"):
        #return
    
    input = Input(shape=DN_INPUT_SHAPE)
    
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation="softmax", name="pi")(x)
    
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name="v")(v)
    
    model = Model(inputs=input, outputs=[p,v])
    
    #os.makedirs("./model/", exist_ok=True)
    #model.save("./model/best.h5")
    
    #K.clear_session()
    #del model
    return model

def make_model():
    model = dual_network()
    #model.save("model_best_4.h5")
    
#if __name__ == "__main__":
    #make_model()
