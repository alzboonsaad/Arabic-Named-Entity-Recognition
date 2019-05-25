#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Input
from keras.models import Model
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
import tensorflow as tf
from keras import backend as K



if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)

    S = "Author's : Sa'ad A. Alzboon && Saja Khaled Tawalbeh"
    print(S + "\n")

# Read training data...

data = pd.read_csv(".../FullDataForTrain.csv", encoding='utf-16', error_bad_lines=False)
data = data.fillna(method="ffill")

words = list(set(TRdata["word"].values))
TAG = list(set(TRdata["Label"].values))

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, v, G, n, P, T) for w, p, v, G, n, P, T in zip(
            s["word"].values.tolist(),
            s["pos"].values.tolist(),
            s["vox"].values.tolist(),
            s["gen"].values.tolist(),
            s["num"].values.tolist(),
            s["per"].values.tolist(),
            s["Label"].values.tolist())]
        self.grouped = self.data.groupby("Sentence_ID").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["SENT: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

sentences = getter.sentences


max_len = 100
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(TAG)}


X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

y = [[tag2idx[w[6]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

y = [to_categorical(i, num_classes=n_tag) for i in y]

# Spliting the data into training && testing, with "0.2" test size...

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False) #0.19945  

# Print the shape of training && testing data after spliting...

print("x_train shape:", X_tr.shape)
print("x_test shape:", X_te.shape)
print("y_train", len(y_tr))
print("y_test", len(y_te))

start_train = datetime.now()
print(start_train)

# Build the "Bi-LSTM-CRF" model with (50-dim) embedding and (300) hidden layer, using RMSprop optimizer...

input = Input(shape=(max_len, ))
model = Embedding(input_dim=n_words + 1, output_dim=50, input_length=max_len, mask_zero=True)(input)  # 50-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
model = Dense(150, activation="relu")(model)  # a dense (Hidden) layer as suggested by neuralNer
model = Dropout(0.1, noise_shape=None, seed=None)(model)
model = Dense(150, activation="relu")(model)  # a dense (Hidden) layer as suggested by neuralNer
crf = CRF(n_tag)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
Rmsprop = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer="Rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

# Fit the "Bi-LSTM-CRF" model...

history = model.fit(X_tr, np.array(y_tr), batch_size=100, epochs=10, validation_split=0.2, verbose=1)
hist = pd.DataFrame(history.history)

# Plot the accuracy && loss of the model...
plt.style.use("ggplot")
plt.figure(figsize=(10, 10))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()

end_train = datetime.now()
print(end_train)
print('Train time : {}'.format(end_train - start_train))


start_test = datetime.now()
print(start_test)

# Make prediction...

print("{}||{}||{}".format("Word"+"\t", "True"+"\t", "Pred"))
print(30 * "=")

for i in range(len(X_te)):
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_tr[i], -1)
    for w, t, pred in zip(X_te[i], true, p[0]):
        if w != 0:
            print("{},\t{},\t{},".format(words[w], TAG[t], TAG[pred]))
            
print("End TEST")

end_time = datetime.now()
print(end_time)
print('Execution Time: (OverAll Time) {}'.format(end_time - start_time))
