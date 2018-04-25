import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()


def my_py_func(x):
    x = tf.matmul(x, x)  # You can use tf ops
    print(x)  # but it's eager!
    return x


with tf.Session() as sess:
    x = tf.placeholder(dtype=tf.float32)
    # Call eager function in graph!
    pf = tfe.py_func(my_py_func, [x], tf.float32)
    sess.run(pf, feed_dict={x: [[2.0]]})


