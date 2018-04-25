import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()


# Summaries and TensorBoard
writer = tf.contrib.summary.create_file_writer(logdir)
global_step = tf.train.get_or_create_global_step()  # return global step var

writer.set_as_default()

for _ in range(iterations):
  global_step.assign_add(1)
  # Must include a record_summaries method
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # your model code goes here
    tf.contrib.summary.scalar('loss', loss)