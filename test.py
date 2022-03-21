import pathlib
import os
from bert_utils import get_ocr_words
from transformers import AutoTokenizer,TFAutoModel
import numpy as np
# import torch

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    # label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    # all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return label_names

# dataset_dir = "/home/sq/data/dateset/rp2k_new/all/train/"
# def generator():
#
#     image_paths, image_labels = get_images_and_labels(dataset_dir)
#     return image_paths, image_labels


def get_bert(word):
    path = "/home/sq/bert_base_uncased"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = TFAutoModel.from_pretrained(path)

    word_k = [word]
    inputs = tokenizer(word_k, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
    x_train = model(inputs)
    return x_train[1][0]

# 生成单独的ocr文件
def ocrfile():
    # dataset_dir = "/home/sq/makefile/total_ocr/"
    dataset_dir = "/home/sq/data/dateset/rp2k_new/all/train/"
    all_image_labels = get_images_and_labels(dataset_dir)
    index = 1
    label_index = 1

    for each_label in all_image_labels:
        if label_index != 362:
            label_index += 1
            continue
        new_root = dataset_dir+str(each_label)
        file = os.listdir(new_root) # 某个label文件夹下的所有文件名称
        new_txt_root = "/home/sq/makefile/total_ocr/" + str(each_label)
        if not os.path.exists(new_txt_root):
            os.makedirs(new_txt_root) # 创建新label
        else:
            print(new_root)
            print("文件夹存在")

        for each_file in file:
            base_name = os.path.splitext(each_file)[0]
            txt_filepath = new_txt_root+"/"+base_name+".txt"
            file = open(txt_filepath, 'w')


            word = get_ocr_words(dataset_dir+each_label+"/"+each_file)
            if len(word)!=0:
                file.write(word)
            file.close()

            index+=1

        print("label:{} have done.{}/2332".format(each_label,label_index))
        label_index+=1
        break

    print(index)

def wordsvecfile():
    # dataset_dir = "/home/sq/makefile/ocr/"
    dataset_dir = "/home/sq/makefile/testocr/"
    all_image_labels = get_images_and_labels(dataset_dir)
    index = 1
    label_index = 1
    for each_label in all_image_labels:
        new_root = dataset_dir + str(each_label)
        file = os.listdir(new_root)  # 某个label文件夹下的所有文件名称

        new_txt_root = "/home/sq/makefile/testwordvecs/" + str(each_label)
        if not os.path.exists(new_txt_root):
            os.makedirs(new_txt_root) # 创建新label

        for each_file in file:
            base_name = os.path.splitext(each_file)[0]
            txt_filepath = new_txt_root + "/" + base_name + ".npz"

            ocr_word = open(dataset_dir + each_label + "/" + each_file, 'r')
            word = ocr_word.read()
            ocr_word.close()
            wordvec = get_bert(word)
            # torch.save(wordvec, new_txt_root + "/" + base_name + ".pt")
            # np.savetxt(txt_filepath, wordvec.numpy())
            np.savez(txt_filepath, wordvec.numpy())
            # print(np.loadtxt(txt_filepath))
            # file.write(wordvec.numpy())

            index += 1

        print("label:{} have done.{}/2332".format(each_label, label_index))
        label_index += 1

    print(index)

# D = np.load("/home/sq/makefile/wordvecs/555（冰炫）/files.npz")
# print(D['arr_0'])
# print(torch.from_numpy(D['arr_0']))
# wordsvecfile()
ocrfile()