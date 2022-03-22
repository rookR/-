import pandas as pd
import jieba
import math
import pathlib
import os
import numpy as np

def shengchengwenjian(datapath, outputpath):
    # 创建一个空的Dataframe
    result = pd.DataFrame()
    #将计算结果逐行插入result,注意变量要用[]括起来,同时ignore_index=True，否则会报错，ValueError: If using all scalar values, you must pass an index
    # datapath = "/home/sq/makefile/total_ocr"
    for fname in pathlib.Path(datapath).glob('*/'):
        print(fname)
        index_name = os.path.basename(fname)
        for fname_in in pathlib.Path(fname).glob('*/'):
            f = open(fname_in, encoding="utf-8")
            str = f.read()
            if len(str)==0:
                continue
            result=result.append(pd.DataFrame({'type':[index_name],'name':[str]}),ignore_index=True)
    # outputpath = "C:\\Users\SQ\Desktop\实验文件\实验结果\word\不筛除空字符\\testoutput.csv"
    result.to_csv(outputpath, encoding='utf_8_sig', sep=',', index=True, header=True)

# shengchengwenjian('/home/sq/makefile/total_ocr', '/home/sq/pycharm_projects/transformers-master/src/transformers/models/vit/total_ocr.csv')
# shengchengwenjian('/home/sq/makefile/total_testocr', '/home/sq/pycharm_projects/transformers-master/src/transformers/models/vit/total_testocr.csv')
from tensorflow.keras.preprocessing import sequence
# def tokenizer(texts, word_index):
#     data = []
#     for sentence in texts:
#         new_txt = []
#         for word in sentence:
#             try:
#                 new_txt.append(word_index[word])  # 把句子中的 词语转化为index
#             except:
#                 new_txt.append(0)
#         data.append(new_txt)
#
#     texts = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
#     return texts



import tensorflow
tensorflow.compat.v1.disable_eager_execution()
tensorflow.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
# gpus = tensorflow.config.experimental.list_physical_devices('GPU')
# 对需要进行限制的GPU进行设置
# tensorflow.config.experimental.set_virtual_device_configuration(gpus[0],[tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=13312)])
from tensorflow.keras.layers import Input, Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv1D,MaxPooling1D,concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding


# train_df = pd.read_csv("C:\\Users\SQ\Desktop\实验文件\实验结果\word\不筛除空字符\output.csv" ,usecols=['type','name'])
# train_df = pd.read_csv("C:\\Users\SQ\Desktop\\ttttest.csv" ,usecols=['type','name'])

# result = pd.DataFrame()
# result=result.append(pd.DataFrame({'type':["555'sb'"],'name':["吸烟有害健康本公司提示请勿在禁烟场所收烟"]}),ignore_index=True)
# result=result.append(pd.DataFrame({'type':["555'sb'"],'name':["吸烟有害健在禁烟场所收烟"]}),ignore_index=True)
# result=result.append(pd.DataFrame({'type':["1"],'name':["禁烟场所收烟"]}),ignore_index=True)
# result=result.append(pd.DataFrame({'type':["2"],'name':["场所收烟"]}),ignore_index=True)
# train_df = result.applymap(lambda x: x if str(x) != 'nan' else " ")
# test_df = result.applymap(lambda x: x if str(x) != 'nan' else " ")

train_df = pd.read_csv("/home/sq/pycharm_projects/transformers-master/src/transformers/models/vit/total_ocr.csv" ,usecols=['type','name'])
test_df = pd.read_csv("/home/sq/pycharm_projects/transformers-master/src/transformers/models/vit/total_testocr.csv" ,usecols=['type','name'])
train_df = train_df.applymap(lambda x: x if str(x) != 'nan' else " ")
test_df = test_df.applymap(lambda x: x if str(x) != 'nan' else " ")


cutWords_list = [] # [['1','2'],['3','4','5']]
content_series = pd.concat([train_df['name'],test_df['name']],axis=0,join='inner',ignore_index=True) # DataFrame
len_list = []
for i in range(len(content_series)):
    content = content_series.iloc[i]
    cutWords = [k for k in jieba.cut(content, True)]
    len_list.append(len(cutWords)) # 每行词的个数
    cutWords_list.append(cutWords)


content_series_train = train_df['name']
len_list = []
cutWords_list_train = [] # [['1','2'],['3','4','5']]
for i in range(len(content_series_train)):
    content = content_series_train.iloc[i]
    cutWords = [k for k in jieba.cut(content, True)]
    len_list.append(len(cutWords)) # 每行词的个数
    cutWords_list_train.append(cutWords)


content_series_test = test_df['name']
cutWords_list_test = [] # [['1','2'],['3','4','5']]
for i in range(len(content_series_test)):
    content = content_series_test.iloc[i]
    cutWords = [k for k in jieba.cut(content, True)]
    len_list.append(len(cutWords)) # 每行词的个数
    cutWords_list_test.append(cutWords)


MAX_SEQUENCE_LENGTH = max(len_list)

# 调用gensim.models库中的Word2Vec类实例化模型对象
from gensim.models import Word2Vec
word2vec_model = Word2Vec(cutWords_list, vector_size=161, min_count=0) # word2vec是对两个一起编码
# vocab_list_2 = [word for word, Vocab in word2vec_model.wv.vocab.items()] # ['1','2','3','4','5']
vocab_list_2 = [word for word in word2vec_model.wv.key_to_index.keys()]
# gensim版本要低于4

# 生成特征矩阵，包含所有内容。保存特征矩阵
embeddings_matrix = np.zeros((len(vocab_list_2)+1, word2vec_model.vector_size)) # 空
word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
word_vector = {} # 初始化`[word : vector]`字典

for i in range(len(vocab_list_2)):
    cutWords = vocab_list_2[i]
    word_index[cutWords] = i + 1 # 词语：序号 这里是一个单词
    word_vector[cutWords] = word2vec_model.wv[cutWords] # 词语：词向量
    embeddings_matrix[i + 1] = word2vec_model.wv[cutWords]  # 词向量矩阵


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cutWords_list) # 合并
vocab = tokenizer.word_index

trainID = tokenizer.texts_to_sequences(cutWords_list_train)
testID = tokenizer.texts_to_sequences(cutWords_list_test)
trainSeq = pad_sequences(trainID, maxlen = MAX_SEQUENCE_LENGTH)
testSeq = pad_sequences(testID, maxlen = MAX_SEQUENCE_LENGTH)


from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
y = labelEncoder.fit(pd.concat([train_df['type'],test_df['type']], axis=0,join='inner',ignore_index=True))
trainLabel = y.transform(train_df['type'])
testLabel = y.transform(test_df['type'])

from tensorflow.keras.utils import to_categorical
trainCate = to_categorical(trainLabel,num_classes=2332)
testCate = to_categorical(testLabel,num_classes=2332)


label_num = 2332

# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def train_mac_lstm(embeddings_matrix,x_train,y_train,x_test,y_test, label_num):
    print('定义Keras Model...')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='sequence_input_later') # (None,161)

    embedding_layer = Embedding(len(embeddings_matrix),
                                161,
                                weights=[embeddings_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH-5, 3, padding="same"))
    model.add(Conv1D(128, 3, padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(label_num, activation='softmax'))

    # from tensorflow.keras.models import load_model
    # model = load_model("C:\\Users\SQ\Desktop\实验文件\实验结果\word\不筛除空字符\my_model.h5")


    # model.summary()
    from tensorflow.keras.optimizers import Adam
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])
    history = model.fit(x_train, y_train, validation_split=0.2,epochs=15, batch_size=64)
    model.save("/home/sq/pycharm_projects/transformers-master/src/transformers/models/vit/mymodel.h5")
    result = model.predict(x_test)
    y_pred = np.argmax(result, axis=1) # 预测值
    score = model.evaluate(x_test,y_test, batch_size=64)
    print(score)


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','valid'],loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()



    # cmtestlabel = np.argmax(y_test, axis=1) # 真实值
    # # 混淆矩阵
    # c = tensorflow.math.confusion_matrix(cmtestlabel.tolist(), y_pred.tolist(),num_classes = 2332)
    # # print(c.shape)
    # label = (labelEncoder.classes_).tolist()
    # c = c.numpy()
    # for i in range(2332):
    #     D = c[i, :]
    #     nonz = np.nonzero(D)
    #     if np.count_nonzero(nonz):  # 非空
    #         print("{}:".format(label[i]), end=' ')
    #         for j in range(np.count_nonzero(nonz)):
    #             print('{}({}))'.format(label[nonz[0][j]], c[i][nonz[0][j]]), end=' ')
    #         print()


    # cmatrix = pd.DataFrame(c.numpy(),columns=(labelEncoder.classes_).tolist(),index=(labelEncoder.classes_).tolist())
    # cmatrix = pd.DataFrame(c.numpy())


    # from pandas.plotting import table
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文字体
    # fig = plt.figure(figsize=(15, 15), dpi=1400)  # dpi表示清晰度
    # ax = fig.add_subplot(111, frame_on=False)
    # ax.xaxis.set_visible(False)  # hide the x axis
    # ax.yaxis.set_visible(False)  # hide the y axis
    # table(ax, cmatrix, loc='center')  # 将df换成需要保存的dataframe即可
    # plt.savefig("C:\\Users\SQ\Desktop\实验文件\实验结果\word\不筛除空字符\out.jpg")



train_mac_lstm(embeddings_matrix, trainSeq, trainCate, testSeq, testCate, label_num)