import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1.Applying arbitrary Python logic with tf.py_func()
# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
    image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    return image_decoded, label


# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


filenames = tf.constant(["./image/ad_0.jpg", "./image/benben02.jpg", "./image/lana_0.jpg"])
labels = [0, 37, 29]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)

iterator = dataset.make_one_shot_iterator()
next_example, next_label = iterator.get_next()
