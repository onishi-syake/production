from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 
from data_arg import perfect_gen
import tensorflow as tf
from pathlib import Path
import numpy as np
import pickle

RN_EPOCHS = 30

def load_data():
    history_path = sorted(Path("./data").glob("*.history"))[-1]
    with history_path.open(mode="rb") as f:
        return pickle.load(f)
    
def train_network(history):
    #history = load_data()
    xs, y_policies, y_values =  zip(*history)
    
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    xs = tf.convert_to_tensor(xs, tf.float32)
    xs = perfect_gen(xs)
    y_policies = tf.constant(y_policies, tf.float32)
    y_pass = y_policies[:, a*b]
    y_pass = tf.tile(y_pass, [8])
    y_pass = tf.expand_dims(y_pass, 1)
    y_policies = tf.reshape(y_policies[:, :a*b], [y_policies.shape[0], a, b])
    y_policies = tf.expand_dims(y_policies, 3)
    y_policies = perfect_gen(y_policies)
    y_policies = tf.reshape(y_policies, [y_policies.shape[0], a*b])
    y_policies = tf.concat([y_policies, y_pass], 1)
    y_values = tf.constant(y_values)
    y_values = tf.tile(y_values, [8])
    
    model = load_model("model_best_3.h5")
    
    model.compile(loss=["categorical_crossentropy", "mse"], optimizer="adam")
    
    def step_decay(epoch):
        x = 0.001
        if epoch >= 10: x = 0.0005
        if epoch >= 20: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)
    
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
            print("\rTrain {}/{}".format(epoch + 1,RN_EPOCHS), end=""))
        
    model.fit(xs, [y_policies, y_values], batch_size=1024, epochs=RN_EPOCHS,
              verbose=1, callbacks=[lr_decay, print_callback])
    print("")
    
    model.save("model_best_3.h5")
    
    K.clear_session()
    del model
    
if __name__ == "__main__":
    train_network()