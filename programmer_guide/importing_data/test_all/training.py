import tensorflow as tf
import load_data
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 2
MAX_STEP = 5
learning_rate = 0.00001
img_height = 300
img_width = 300
num_epochs = 1


def train():
    filenames = tf.placeholder(tf.string, shape=[None])
    training_filenames = ["./train.records"]
    validation_filenames = ["./train.records"]
    iterator = load_data.read_dataset(filenames, img_height, img_width, BATCH_SIZE, num_epochs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        tra_img, tra_label = iterator.get_next()
        try:
            for step in range(MAX_STEP):
                tra_img, tra_label = sess.run([tra_img, tra_label])
                for j in range(BATCH_SIZE):
                    print(j, tra_label[j])
                    print(type(tra_label))
                    print("-----------------------")
                    plt.imshow(tra_img[j, :, :, :])
                    plt.show()
        except tf.errors.OutOfRangeError:
            print('done!')


if __name__ == '__main__':
    train()


