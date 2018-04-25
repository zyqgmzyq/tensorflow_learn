import numpy as np
import tensorflow as tf
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_dir = "./list.txt"
train_dir = "./image/"
# train_dir = "/home/mvl/Dataset/ColorConstancy/Cube/JPG/"
# label_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt.txt"
# img_height = 3456
# img_width = 5184

tf_file = "train.records"
# tf_file1 = "train1.tfrecords"
# tf_file2 = "train2.tfrecords"
# tf_file3 = "train3.tfrecords"
# train1_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg1/"
# train2_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg2/"
# train3_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg3/"
# label1_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt1.txt"
# label2_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt2.txt"
# label3_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt3.txt"


def write(filename, train_name, label_name, img_width, img_height):
    writer = tf.python_io.TFRecordWriter(filename)
    labels = np.zeros(3, dtype=np.float32)
    l = os.listdir(train_name)
    # 排序
    l = sorted(l, key=lambda name: int(name[:-4]))

    for file, line in zip(l, open(label_name)):
        ll = line.split(' ')
        labels[0] = ll[0]
        labels[1] = ll[1]
        labels[2] = ll[2]
        print(labels)
        if file.endswith(".jpg"):
            file_path = train_dir + file
            print(file_path)
            img = Image.open(file_path)
            # 处理图片的大小
            img = img.resize((img_width, img_height))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                # 图片对应单个结果
                # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[sort(file[:file.rindex("_")])])),
                # 图片对应多个结果
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


def normalize(image, means, stds):
    """normalize a image by substract mean and std
    Args :
        image: 3-D image
        means: list of C mean values
        stds: list of C std values
    Returns:
        3-D normalized image

    for imagenet pretrained model:
        means = [0.485, 0.456, 0.406] R G B
        std = [0.229, 0.224, 0.225] R G B
    """
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
        channels[i] /= stds[i]
    return tf.concat(axis=2, values=channels)


def read_dataset(filenames, img_height, img_width, batch_size, num_epochs):
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {'label': tf.FixedLenFeature([3], tf.float32),
                  'img_raw': tf.FixedLenFeature([], tf.string),}
        parsed = tf.parse_single_example(record, features)
        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["img_raw"], tf.uint8)
        image = tf.reshape(image, [img_height, img_width, 3])

        # normalize
        image = tf.cast(image, tf.float32) * (1. / 255)
        # image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # labels
        label = tf.cast(parsed["label"], tf.float32)
        label = tf.nn.l2_normalize(label)
        return image, label

    # Parse the record into tensors.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    # Repeat the input indefinitely.
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator
    # features, labels = iterator.get_next()
    # return features, labels


if __name__ == '__main__':
    write(tf_file, train_dir, label_dir, 300, 300)
