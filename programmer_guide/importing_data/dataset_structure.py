import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print("data1\n", data1.output_types)  # ==> "tf.float32"
print(data1.output_shapes)  # ==> "(10,)"

data2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),
                                            tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print("data2\n", data2.output_types)  # ==> "(tf.float32, tf.int32)"
print(data2.output_shapes)  # ==> "((), (100,))"

data3 = tf.data.Dataset.zip((data1, data2))
print("data3:\n", data3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(data3.output_shapes)  # ==> "(10, ((), (100,)))"


data = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(data.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(data.output_shapes)  # ==> "{'a': (), 'b': (100,)}"





