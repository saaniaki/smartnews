import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
import sklearn.datasets as skds
from flask import Flask, json, request, jsonify
from flask_cors import cross_origin

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

encoder = LabelBinarizer()
encoder.fit(train_tags)

# Load the model
filepath = './saved_model'
model = load_model(filepath, compile = True)


# Server

api = Flask(__name__)

@api.route('/analyze', methods=['POST', 'OPTIONS'])
@cross_origin()
def analyze_news():
  print(request)
  if request.get_json() is not None:
    txt = request.get_json()['text']
    text_labels = encoder.classes_
    input_val = tokenizer.texts_to_matrix([txt], mode='tfidf')
    prediction = model.predict(np.array([input_val[0]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print("Predicted label: " + predicted_label)
    return jsonify(tags=predicted_label)
  return jsonify(tags="")

if __name__ == '__main__':
    api.run()
