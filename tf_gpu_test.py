import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy
import pathlib
import random
# image_raw_data = tf.compat.v1.gfile.FastGFile("截屏2019-12-09下午7.50.51.png","rb").read()
# with tf.compat.v1.Session() as sess:
#     # 解压jpeg格式的图像文件从而得到图像对应的三维矩阵
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # img_data = tf.image.decode_png(image_raw_data)
#
#     img = cv2.cvtColor(numpy.asarray(img_data.eval()), cv2.COLOR_RGB2BGR)
#     cv2.imshow("res", img)
#     cv2.waitKey(0)
#     # plt.imshow(img_data.eval())
#     # plt.show()

def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict
#
def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]
    return all_image_path, all_image_label
#
def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0.))):
        value = value.numpy()   # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
#
def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    # shuffle the dict
    image_paths_and_labels_dict = shuffle_dict(image_paths_and_labels_dict)
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, label in image_paths_and_labels_dict.items():
            print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
#
# def get_the_length_of_dataset(dataset):
#     count = 0
#     for i in dataset:
#         count += 1
#     return count
#
# def _parse_image_function(example_proto):
#     # Parse the input tf.Example proto.
#     return tf.io.parse_single_example(example_proto, {
#         'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
#         'image_raw': tf.io.FixedLenFeature([], tf.dtypes.string),
#     })
#
# def get_parsed_dataset(tfrecord_name):
#     raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
#     # parsed_dataset = raw_dataset.map(_parse_image_function,num_parallel_calls=tf.data.AUTOTUNE)
#     parsed_dataset = raw_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     return parsed_dataset
#
# def generate_datasets():
#     train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
#     train_count = get_the_length_of_dataset(train_dataset)
#     train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
#     return train_dataset
#
#
# def load_and_preprocess_image(image_raw, data_augmentation=False):
#     # decode
#     image_tensor = tf.io.decode_image(contents=image_raw, channels=3, dtype=tf.dtypes.float32)  # 得到图像的像素值
#     image = tf.image.resize(image_tensor, [224, 224])
#     return image
#
# def process_features(features, data_augmentation):
#     image_raw = features['image_raw'].numpy()
#     image_tensor_list = []
#     for image in image_raw:
#         image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
#         image_tensor_list.append(image_tensor)
#     images = tf.stack(image_tensor_list, axis=0)
#     labels = features['label'].numpy()
#     return images, labels

BATCH_SIZE = 2
train_dir = "C:\\Users\SQ\Desktop\泸州老窖精品头曲组合装\\"
train_tfrecord = "C:\\Users\SQ\Desktop\\train.tfrecords"
dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord) # 数据集转化成tfrecords


# train_dataset = generate_datasets() # <BatchDataset shapes: {image_raw: (None,), label: (None,)}, types: {image_raw: tf.string, label: tf.int64}>
# for features in train_dataset:
#     images, labels = process_features(features, data_augmentation=False)