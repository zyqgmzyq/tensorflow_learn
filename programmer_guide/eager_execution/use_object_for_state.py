import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

# 1.Variables are objects

v = tfe.Variable(tf.random_normal([1000, 1000]))


# 2.Object-based saving
x = tfe.Variable(10.)
checkpoint = tfe.Checkpoint(x=x)  # save as "x"
x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')
x.assign(11.)  # Change the variable after saving.
# Restore values from the checkpoint
checkpoint.restore(save_path)

print(x)  # => 2.0


# save and load models
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result


model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = "./model_dir"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

