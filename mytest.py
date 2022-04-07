from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path
import re
import sys
import tarfile
import copy
import sys
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


from text_recognition import TextRecognition
from text_detection import TextDetection

from util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw

# !flask/bin/python
from flask import Flask, jsonify, flash, Response
from flask import make_response
from flask import request, render_template
from flask_bootstrap import Bootstrap
from flask import redirect, url_for
from flask import send_from_directory

from werkzeug.utils import secure_filename
from subprocess import call

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'image'
VIDEO_FOLDER = r'video'
FOND_PATH = 'STXINWEI.TTF'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4','avi'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])

def init_ocr_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    # recognition_checkpoint='/data/zhangjinjin/icdar2019/LSVT/full/recognition/checkpoint_3x_single_gpu/OCR-443861'
    # recognition_pb = './checkpoint/text_recognition_5435.pb' #
    recognition_pb = './checkpoint/text_recognition.pb'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    with tf.device('/gpu:2'):
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    label_dict = np.load('./reverse_label_dict_with_rects.npy')[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict

# ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()

def init_detec_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb'
    with tf.device('/gpu:0'):
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
    label_dict = np.load('./reverse_label_dict_with_rects.npy')[()]
    return detection_model, label_dict
def detec(image, detection_model, label_dict):
    vis_image = bgr_image = cv2.imread(image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)
    # print(r_boxes)
    return r_boxes



from functools import reduce
import operator
import math


def order_points(pts):
    def centeroidpython(pts):
        x, y = zip(*pts)
        l = len(x)
        return sum(x) / l, sum(y) / l

    centroid_x, centroid_y = centeroidpython(pts)
    pts_sorted = sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x)))
    return pts_sorted


def draw_annotation(image, points, label, horizon=True, vis_color=(255,0,0)):#(30,255,255)
    points = np.asarray(points)
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)

    image = Image.fromarray(image)
    width, height = image.size
    fond_size = int(max(height, width)*0.015)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    points = order_points(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT)
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)
    return image


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox


def detection(image, detection_model, recognition_model, label_dict, it_is_video=False):
    # bgr_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    vis_image = bgr_image = cv2.imread(image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)
    json_result = []

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            # print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        # print(image_padded.shape)

        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        #print(''.join(results))
        # print(probs)

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        pts = list(ccw_polygon.exterior.coords)[:-1]
        vis_image = draw_annotation(vis_image, pts, ''.join([]))
        json_result.append(''.join(results))
    fname = os.path.basename(image)
    save_path = 'static/image/' + uuid.uuid4().hex + fname[fname.index(".") :]
    cv2.imwrite(save_path, vis_image)
    save_path = os.path.join('/static/image/', os.path.basename(save_path))
    return json_result, save_path

ocr_detection_model, ocr_label_dict = init_detec_model()
class OCR:
    def result(self, image):
        # 返回数组[{}, {}, {}]
        return detec(image, ocr_detection_model, ocr_label_dict)

def cropimgs(img_path, new_txt_root, base_name):
    ocr = OCR()
    result = ocr.result(img_path)
    it = iter(result)
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(result)):
        line = next(it) # line = [[ 77, 511],[ 72, 418],[423, 399],[428, 493]]

        left = min(line[0][0], line[1][0], line[2][0], line[3][0])
        right = max(line[0][0], line[1][0], line[2][0], line[3][0])
        upper = min(line[0][1], line[1][1], line[2][1], line[3][1])
        lower = max(line[0][1], line[1][1], line[2][1], line[3][1])

        # box = (left, upper, right, lower)
        # region = img.crop(box) # (left, upper, right, lower)
        # region.save(new_txt_root+"/"+base_name+'_'+str(i+1)+'.png') 分开保存所有检测图像

        upper = max(upper, 0)
        lower = max(lower, 0)
        left = max(left, 0)
        right = max(right, 0)
        mask[int(upper):int(lower), int(left):int(right)] = 255
    res = cv2.bitwise_and(img, mask)
    # res.imwrite(new_txt_root + "/" + base_name + '.png')
    cv2.imwrite(new_txt_root + "/" + base_name + '.png',res)


import pathlib
def get_images_and_labels(data_root_dir):
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    label_names = sorted(item.name for item in data_root.glob('*/'))
    return label_names


dataset_dir = "/home/sq/data/dateset/rp2k_new/all/test/"
all_image_labels = get_images_and_labels(dataset_dir)
index = 1
label_index = 1

for each_label in all_image_labels:
    new_root = dataset_dir+str(each_label)
    file = os.listdir(new_root) # 某个label文件夹下的所有文件名称
    new_txt_root = "/home/sq/data/cropimgs/test/" + str(each_label)
    if not os.path.exists(new_txt_root):
        os.makedirs(new_txt_root) # 创建新label
    else:
        print(new_root)
        print("文件夹存在")

    for each_file in file:
        base_name = os.path.splitext(each_file)[0]
        cropimgs(dataset_dir+each_label+"/"+each_file, new_txt_root, base_name)
        index+=1

    print("label:{} have done.{}/2332".format(each_label,label_index))
    label_index+=1

print(index)