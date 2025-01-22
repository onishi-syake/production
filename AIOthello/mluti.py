import time
import os
from multiprocessing import Pool
#import tensorflow as tf
def function(x):
    """実行したい処理"""
    num = 0
    for i in range(100000):
        num += 1
    return x

def multi(n):
    """マルチプロセス"""
    p = Pool(8) # プロセス数
    result = p.map(function, range(n)) # 関数を並列で呼び出す
    p.close()
    return result

def single(n):
    """シングルプロセス"""
    result = [function(x) for x in range(n)]
    return result
'''@tf.function
def gpu(n):
    """シングルプロセス"""
    a=tf.TensorArray(tf.int32,n*100000)
    for x in tf.range(n):
        for i in tf.range(100000):
           a.write(x*100000+i, n+1)        
    return a'''

if __name__ == "__main__":
    start1 = time.time()
    print(multi(1000))
    #    print(x[0][0:10])
    elapsed_time1 = time.time() - start1
    print ("elapsed_time:{0}".format(elapsed_time1) + "[sec]")
    
    start1 = time.time()
    single(1000)
    #    print(x[0][0:10])
    elapsed_time1 = time.time() - start1
    print ("elapsed_time:{0}".format(elapsed_time1) + "[sec]")
    
    