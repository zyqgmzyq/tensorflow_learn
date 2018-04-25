import tensorflow as tf
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1.Decoding image data and resizing it
# Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    # image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_decoded, label


# A vector of filename.
filenames = tf.placeholder(tf.string, shape=[None])
# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([[0, 37, 5], [3, 5 ,6], [4, 5, 6]])
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
iterator = dataset.make_initializable_iterator()
# iterator = dataset.make_one_shot_iterator()

file = ["./image/1.jpg", "./image/benben_1.jpg", "./image/lana_0.jpg"]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer, feed_dict={filenames: file})
    next_example, next_label = iterator.get_next()
    for j in range(3):
        image, label = sess.run([next_example, next_label])
        print(label)
        print(type(label))
        print(type(image))
        print(image)
        print("-----------------------")
        plt.imshow(image)
        plt.show()



