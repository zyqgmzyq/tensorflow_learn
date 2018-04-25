import tensorflow as tf
import load_data
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 1
MAX_STEP = 2
learning_rate = 0.00001
img_height = 300
img_width = 300


def train():
    filenames = tf.placeholder(tf.string, shape=[None])
    training_filenames = ["./train.records"]
    validation_filenames = ["./train.records"]
    iterator = load_data.read_dataset(filenames, img_height, img_width, BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        tra_img, tra_label = iterator.get_next()
        print(type(tra_img))
        try:
            for step in range(MAX_STEP):
                tra_img1, tra_label1 = sess.run([tra_img, tra_label])
                for j in range(BATCH_SIZE):
                    print(step, tra_label1[j])
                    print(type(tra_label1))
                    print("-----------------------")
                    plt.imshow(tra_img1[j, :, :, :])
                    plt.show()
        except tf.errors.OutOfRangeError:
            print('done!')


if __name__ == '__main__':
    train()


