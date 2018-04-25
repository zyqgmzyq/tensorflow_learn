# encoding=utf-8
import os
import tensorflow as tf
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


classes = {'test', 'test1', 'test2'}
# 制作二进制数据


def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = "./image/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_raw = img.tobytes()
            # 将图片转化为原生bytes
            print(img_name, index, img_raw)
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()

data = create_record()

# 读取二进制数据


def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label

if __name__ == '__main__':
    if 0:
        data = create_record("train.tfrecords")
    else:
        img, label = read_and_decode("train.tfrecords")
        print("tengxing", img, label)
        #使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
        # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
        # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
        # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
        # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=4, capacity=2000,
                                                    min_after_dequeue=1000)

        # 初始化所有的op
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # 启动队列
            threads = tf.train.start_queue_runners(sess=sess)
            for i in range(5):
                print(i, img_batch.shape, label_batch)
                val, l = sess.run([img_batch, label_batch])
                # l = to_categorical(l, 12)
                print(i, val.shape, l)



