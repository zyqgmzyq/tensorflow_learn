import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

# 1.basic use
print(tf.executing_eagerly())
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print("a:", a)
# Broadcasting support
b = tf.add(a, 1)
print("b:", b)
# Operator overloading is supported
print("a * b:", a * b)
c = np.multiply(a, b)
print("c:", c)
# Obtain numpy value from a tensor:
print("a.numpy:", a.numpy())


# 2.Eager training


# 2.1 Automatic differentiation
w = tfe.Variable([[1.0]])
with tfe.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, [w])
print("grad:", grad)


# 2.2 an example of tfe.GradientTape  train a simple model
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise


def prediction(input, weight, bias):
  return input * weight + bias


# A loss function using mean-squared error
def loss(weights, biases):
    error = prediction(training_inputs, weights, biases) - training_outputs
    return tf.reduce_mean(tf.square(error))


# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
    with tfe.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value, [weights, biases])


train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))


# 3.Dynamic models
def line_search_step(fn, init_x, rate=1.0):
  with tfe.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad, = tape.gradient(value, [init_x])
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value


# 4.Additional functions to compute gradients
def square(x):
  return tf.multiply(x, x)
grad = tfe.gradients_function(square)
print("square(3.):", square(3.))
print("grad(3.):", grad(3.))

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
print("gradgrad(3.):", gradgrad(3.))

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
print("gradgradgrad(3.) :", gradgradgrad(3.) )

# With flow control:
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)

print("grad(3.):", grad(3.))
print("grad(-3.):", grad(-3.))


# 5.Custom gradients
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn

def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)
# The gradient computation works fine at x = 0.
print("grad_log1pexp(0.) :", grad_log1pexp(0.) ) # => [0.5]
# However, x = 100 fails because of numerical instability.
print("grad_log1pexp(100.) :",grad_log1pexp(100.)) # => [nan]

@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad
grad_log1pexp = tfe.gradients_function(log1pexp)
# As before, the gradient computation works fine at x = 0.
print("grad_log1pexp(0.) :", grad_log1pexp(0.))
# And the gradient computation also works at x = 100.
print("grad_log1pexp(100.)  :", grad_log1pexp(100.)  )