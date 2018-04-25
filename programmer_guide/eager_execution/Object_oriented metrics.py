import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

m = tfe.metrics.Mean("loss")
m(0)
m(5)
print(m.result())  # => 2.5
m([8, 9])
print(m.result())   # => 5.5
