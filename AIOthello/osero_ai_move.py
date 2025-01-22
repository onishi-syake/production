# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:01:03 2021

@author: 84840
"""
roop_num = 25

for i in range(roop_num):#5roop == 1hour+few
    from tri_predict_osero_ai import State
    from tri_predict_osero_ai import BOARD_SIZE, BOARD_LENGTH
    import tensorflow as tf
    import random
    import time
    print(i)
    
    state = State()
    start = time.time()
    while True:
        state.keep()
        state.drop()
        if state.pieces.shape == tf.constant([]).shape:
            break
        state.pieces = state.predicter(1-(i/roop_num)/2)
        print("state.depth",state.depth)
        state.depth += 1
    print("time",time.time()-start)
    #print(tf.transpose(tf.reshape(state.keep_kihu[34, :72], [2, 6, 6]), [1, 2, 0]), state.keep_kihu[34, 72])
    print(state.keep_kihu.shape)
    
    def perfect_gen(train_log):
      l1 = tf.image.flip_left_right(train_log)
      l2 = tf.image.flip_up_down(train_log)
      l3 = tf.image.flip_left_right(l2)
      l4 = tf.image.rot90(train_log)
      l5 = tf.image.flip_left_right(l4)
      l6 = tf.image.flip_up_down(l4)
      l7 = tf.image.flip_left_right(l6)
      return tf.concat([train_log,l1,l2,l3,l4,l5,l6,l7], 0)
    
    
    train_data = state.keep_kihu
    print(train_data.shape)
    random_list = [random.randint(0, train_data.shape[0]-1) for _ in range(train_data.shape[0])]
    random_list = tf.cast(random_list, tf.int32)
    r_train_data = tf.gather(train_data, random_list)
    r_train_log = tf.reshape(r_train_data[:, :BOARD_SIZE*2], [r_train_data.shape[0], 2, BOARD_LENGTH, BOARD_LENGTH])
    r_train_log = tf.transpose(r_train_log, [0, 2, 3, 1])
    #r_train_log = perfect_gen(r_train_log)
    r_train_label = r_train_data[:, BOARD_SIZE*2]
    #r_train_label = tf.tile(r_train_data[:, BOARD_SIZE*2], [8])
    #print(r_train_log.shape, r_train_label.shape)
    model = tf.keras.models.load_model('osero_model_6.h5')
    history = model.fit(r_train_log,r_train_label,epochs=100,validation_split=(0.2),batch_size=1024)
    model.save('osero_model_6.h5')
    
    """if i % 5 == 0:
        train_data = state.keep_kihu
    else:
        train_data = tf.concat([train_data, state.keep_kihu], 0)
    if i % 5 == 4:
        print(train_data.shape)
        random_list = [random.randint(0, train_data.shape[0]-1) for _ in range(train_data.shape[0])]
        random_list = tf.cast(random_list, tf.int32)
        r_train_data = tf.gather(train_data, random_list)
        r_train_log = tf.reshape(r_train_data[:, :BOARD_SIZE*2], [r_train_data.shape[0], 2, BOARD_LENGTH, BOARD_LENGTH])
        r_train_log = tf.transpose(r_train_log, [0, 2, 3, 1])
        #r_train_log = perfect_gen(r_train_log)
        r_train_label = r_train_data[:, BOARD_SIZE*2]
        #r_train_label = tf.tile(r_train_data[:, BOARD_SIZE*2], [8])
        #print(r_train_log.shape, r_train_label.shape)
        model = tf.keras.models.load_model('osero_model_4.h5')
        history = model.fit(r_train_log,r_train_label,epochs=100,validation_split=(0.2),batch_size=1024)
        model.save('osero_model_4.h5')"""
#print(tf.transpose(tf.reshape(state.keep_kihu[60, :128], [2, 8, 8]), [1, 2, 0]))
#print(state.keep_kihu[60, 128])
#print(tf.tile(tf.constant([3, 4]), [8]))
#棋譜とラベルのチェック用
"""a = state.keep_kihu[:, :BOARD_SIZE*2]
c = tf.cast(a, tf.int32)
b = tf.reduce_sum(c, 1)
d = tf.where(b == BOARD_SIZE)
d = tf.reshape(d, [d.shape[0]])
e = tf.gather(state.keep_kihu, d)
z = tf.gather(state.keep_kihu, d-2)
z = tf.cast(z, tf.int32)
f = e[:, :BOARD_SIZE]
f = tf.reduce_sum(f, 1)
g = e[:, BOARD_SIZE:BOARD_SIZE*2]
g = tf.reduce_sum(g, 1)
h = f < g
h = tf.logical_not(h)
h = tf.cast(h, tf.int32)
h = h - 1
h = tf.reshape(h, [h.shape[0]])
i = e[:, BOARD_SIZE*2]
zi = z[:, BOARD_SIZE*2]
i = tf.cast(i, tf.int32)
#i = tf.where(i == 1, 0, i)
#print(h == i)
print(zi == i)



a = r_train_log
print(a.shape)
a = tf.transpose(a, [0, 3, 1, 2])
a = tf.reshape(a, [a.shape[0], BOARD_SIZE*2])
b = tf.cast(a, tf.int32)
b = tf.reduce_sum(b, 1)
print("b.shape", b.shape)
d = tf.where(b == BOARD_SIZE)
d = tf.reshape(d, [d.shape[0]])
e = tf.gather(a, d)
f = e[:, :BOARD_SIZE]
f = tf.reduce_sum(f, 1)
g = e[:, BOARD_SIZE:BOARD_SIZE*2]
g = tf.reduce_sum(g, 1)
print(f.shape, g.shape)
h = f < g
h = tf.logical_not(h)
h = tf.cast(h, tf.int32)
h = h - 1
print("h.shape", h.shape)
h = tf.reshape(h, [h.shape[0]])
i = r_train_label
i = tf.cast(i, tf.int32)
i = tf.gather(i, d)
i = tf.where(i == 1, 0, i)
print(h.shape, i.shape)
print(h == i)"""