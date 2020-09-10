import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, save_model
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import sklearn.datasets as skds
from sklearn.utils import class_weight
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE

# For reproducibility
np.random.seed(1237)

# ------------------------------------------------------------------------
# Source file directory
# downlaod from http://qwone.com/~jason/20Newsgroups/

path_train = "rawdata/20news-bydate-train"

files_train = skds.load_files(path_train, load_content=False)

label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ["filename", "category", "news"]
data_list = []

# Read and add data from file to a list
i = 0
for f in labelled_files:
    data_list.append(
        (f, label_names[label_index[i]], Path(f).read_text(encoding='latin1')))
    i += 1

# We have training data available as dictionary filename, category, data
data = pd.DataFrame.from_records(data_list, columns=data_tags)

# print(data.head())

# 20 news groups
num_labels = 20
vocab_size = 15000
batch_size = 100
num_epochs = 30

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .8)

train_posts = data['news'][:train_size]
train_tags = data['category'][:train_size]
train_files_names = data['filename'][:train_size]

test_posts = data['news'][train_size:]
test_tags = data['category'][train_size:]
test_files_names = data['filename'][train_size:]

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

print(x_train.shape, y_train.shape)


print('Build model...')

#let us build a basic model
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




# EMBEDDING_DIM = 100

# model = Sequential()
# model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
# model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(num_labels, activation='softmax'))

# # try using different optimizers and different optimizer configs
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())



num_epochs =10
batch_size = 128
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_split=0.2)

score, acc = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=2)

print('Test accuracy:', acc)

# Save the model
filepath = './saved_model'
save_model(model, filepath)
