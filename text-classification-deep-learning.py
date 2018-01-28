# coding=utf-8

# 常量
MAX_SEQUENCE_LENGTH = 300  # 每条新闻最大长度
EMBEDDING_DIM = 200  # 词向量空间维度
VALIDATION_SPLIT = 0.20  # 验证集比例
SENTENCE_NUM = 30 # 句子的数目
model_filepath = "./CNN_word2vec_1_layer_model"

from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding,GRU
from keras.optimizers import Adam
from keras import regularizers
import gensim
from time import time
import keras.callbacks
from keras.layers import LSTM,Bidirectional
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# input data
train_filename = "./THUCNews_deal_title/dataset/sentenced_deal_stop_word_train.csv.zip"
test_filename = "./THUCNews_deal_title/dataset/sentenced_deal_stop_word_test.csv.zip"
train_df = pd.read_csv(train_filename,sep='|',compression = 'zip',error_bad_lines=False)
test_df = pd.read_csv(test_filename,sep='|',compression='zip',error_bad_lines=False)
content_df = train_df.append(test_df, ignore_index=True)
# shuffle data
from sklearn.utils import shuffle  
content_df = shuffle(content_df) 

all_texts = content_df['content']
all_labels = content_df['type']
print "新闻文本数量：", len(all_texts), len(all_labels)
print "每类新闻的数量：\n", all_labels.value_counts()

all_texts = all_texts.tolist()
all_labels = all_labels.tolist()

original_labels = list(set(all_labels))
num_labels = len(original_labels)
print "label counts:", num_labels
# one_hot encode label
one_hot = np.zeros((num_labels, num_labels), int)
np.fill_diagonal(one_hot, 1)
label_dict = dict(zip(original_labels, one_hot))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print "Found %s unique tokens." % len(word_index)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.asarray([label_dict[x] for x in all_labels])
print "shape of data tensor:", data.shape
print "shape of label tensor:", labels.shape

# 分割训练集和测试集
p_train = int(len(data) * (1 - VALIDATION_SPLIT))
x_train = data[:p_train]
y_train = labels[:p_train]
x_val = data[p_train:]
y_val = labels[p_train:]
print 'train docs:' + str(len(x_train))
print 'validate docs:' + str(len(x_val))

# 搭建模型
def cnn_model(embedding_layer=None):
	model = Sequential()
	if embedding_layer:
	# embedding layer use pre_trained word2vec model
		model.add(embedding_layer)
	else:
	# random word vector
		model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
	model.add(Conv1D(128, 5, padding='valid', activation='relu'))
	model.add(MaxPooling1D(5))
# 	model.add(Conv1D(128, 5, padding='valid', activation='relu'))
# 	model.add(MaxPooling1D(5))
# 	model.add(Conv1D(128, 5, padding='valid', activation='relu'))
# 	model.add(MaxPooling1D(5))
 	model.add(Flatten())
	return model

def lstm_model(embedding_layer=None):
	model = Sequential()
	if embedding_layer:
	# embedding layer use pre_trained word2vec model
		model.add(embedding_layer)
	else:
	# random word vector
		model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
	# Native LSTM
	model.add(LSTM(200,dropout=0.2,recurrent_dropout=0.2))
	return model

def gru_model(embedding_layer=None):
	model = Sequential()
	if embedding_layer:
	# embedding layer use pre_trained word2vec model
		model.add(embedding_layer)
	else:
	# random word vector
		model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
	# GRU 
	model.add(GRU(200,dropout=0.2,recurrent_dropout=0.2))
	return model

def bidirectional_lstm_model(embedding_layer=None):
	model = Sequential()
	if embedding_layer:
	# embedding layer use pre_trained word2vec model
		model.add(embedding_layer)
	else:
	# random word vector
		model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
	# Bidirection LSTM
	model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2))
	return model

def cnn_lstm_model(embedding_layer=None):
	model = Sequential()
	if embedding_layer:
	# embedding layer use pre_trained word2vec model
		model.add(embedding_layer)
	else:
	# random word vector
		model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
	model.add(Conv1D(128, 5, padding='valid', activation='relu'))
	model.add(MaxPooling1D(5))
	# model.add(Dropout(0.5))
	# model.add(GRU(128,dropout=0.2,recurrent_dropout=0.1,return_sequences = True))
	model.add(GRU(128,dropout=0.2,recurrent_dropout=0.1))
	return model


# load word2vec model
word2vec_model_file = "/home/zwei/workspace/nlp_study/word2vec_wiki_study/word2vec_train_wiki/WIKI_word2vec_model/word2vec_skip_ngram_200_10_5_model.bin"
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_file, binary=True)
# print "test word 奥巴马:",word2vec_model['奥巴马'.decode('utf-8')]

# construct embedding layer
embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))

no_word_file = open("no_word_file.txt",'w')

for word, i in word_index.items():
    if word.decode('utf-8') in word2vec_model:
        embedding_matrix[i] = np.asarray(word2vec_model[word.decode('utf-8')], dtype='float32')
    else:
#        print "word not found in word2vec:", word
        no_word_file.write(word+"\n")        
        embedding_matrix[i] = np.random.random(size=EMBEDDING_DIM)

no_word_file.close()

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# CNN + word2vec
# model = cnn_model(embedding_layer)

# CNN + radom word vector
# model = cnn_model()

# LSTM 
# model = lstm_model(embedding_layer) 
 
# GRU 
model = gru_model(embedding_layer)

# CNN + LSTM
# model = cnn_lstm_model(embedding_layer)

# common
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1], activation='softmax',kernel_regularizer=regularizers.l2(0.1)))
model.summary()

# adam = Adam(lr=0.01)

# tensorboard
tensorboard = keras.callbacks.TensorBoard(log_dir="lstm_word2vec_log_THUCNews_deal_title/{}".format(time()))

# function
from keras import backend as K
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=256,callbacks=[tensorboard], validation_split=0.1)

# model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))
# score = model.evaluate(x_val, y_val, batch_size=128)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# # sklearn matrics 
# from sklearn.metrics import confusion_matrix
# y_pred = model.predict(x_val,batch_size=64)
# y_pred_label = [c.index(max(c)) for c in y_pred.tolist()]
# y_true_label = [c.index(max(c)) for c in y_val.tolist()]
# y_pred_label = [original_labels[i] for i in y_pred_label]
# y_true_label = [original_labels[i] for i in y_true_label]
# matrix = confusion_matrix(y_true_label, y_pred_label,original_labels)
# # matplotlib
# import matplotlib.pyplot as plt
# plt.matshow(matrix)
# plt.colorbar()
# plt.xlabel('Prediction')
# plt.ylabel('True')
# plt.xticks(matrix[1],label_dict.keys())
# plt.yticks(matrix[1],label_dict.keys())
# # plt.show()
# plt.savefig("confusion_matrix.jpg")
# 
# # classification_report
# from sklearn.metrics import classification_report  
# print "classification_report(left: labels):"  
# for key,value in label_dict.iteritems():
#     print "dict[%s]="%key,value
# print classification_report(y_val, y_pred) 
# 
# 
# show model
# from keras.utils import plot_model
# 
# model.save_weights(model_filepath)
# plot_model(model, to_file='model.png')
