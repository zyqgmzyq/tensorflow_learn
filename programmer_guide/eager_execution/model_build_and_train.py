import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

# 1.Build a model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])


class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()


# 2.Train a model
# Create a tensor representing a blank image
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)
result = model(batch)
print("result:", result)


# 3.This example uses the dataset.py module from the TensorFlow MNIST example
import dataset  # download dataset.py file
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)


def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = tfe.Iterator(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

# Training loop
for (i, (x, y)) in enumerate(tfe.Iterator(dataset_train)):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))

