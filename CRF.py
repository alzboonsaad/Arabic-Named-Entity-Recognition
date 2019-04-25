#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import sequence_accuracy_score
from sklearn.model_selection import cross_val_predict

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(start_time)

    S = "Author's : Sa'ad A. Alzboon && Saja Khaled Tawalbeh"
    print(S + "\n")

# Read training data....

TRdata = pd.read_csv(".../FullDataForTrain.csv", encoding='utf-16', error_bad_lines=False)  # 12486
TRdata = TRdata.fillna(method="ffill")

# Read testing data...

TEdata = pd.read_csv(".../FullDataForTest.csv", encoding='utf-16', error_bad_lines=False)  # 3109
TEdata = TEdata.fillna(method="ffill")

# Count the number of uniqe words of training data...

words = list(set(TRdata["word"].values))
TR_words = len(words)
print("Number of Different Words :" + "n_words")
print(TR_words)

# Count the number of uniqe "POS-tags" of training data...

POS = list(set(TRdata["pos"].values))
TR_pos = len(POS)
print("Number of Different POS tags :" + "n_pos")
print(TR_pos)

# Count the number of uniqe "Labels" of training data...

TAG = list(set(TRdata["Label"].values))
TR_tag = len(TAG)
print("Number of Different NER tags :" + "n_tag")
print(TR_tag)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, v, G, n, P, T) for w, p, v, G, n, P, T in
                              zip(s["word"].values.tolist(), s["pos"].values.tolist(), s["vox"].values.tolist(),
                                  s["gen"].values.tolist(), s["num"].values.tolist(), s["per"].values.tolist(),
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


TRgetter = SentenceGetter(TRdata)
TEgetter = SentenceGetter(TEdata)
sent = TRgetter.get_next()
print(sent)

TRsentences = TRgetter.sentences
TEsentences = TEgetter.sentences

# Features extraction...

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    vox = sent[i][2]
    gen = sent[i][3]
    num = sent[i][4]
    per = sent[i][5]

    features = {'bias': 1.0, 'word': word, 'postag': postag, 'vox': vox, 'gen': gen, 'num': num, 'per': per, }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        vox1 = sent[i - 1][2]
        gen1 = sent[i - 1][3]
        num1 = sent[i - 1][4]
        per1 = sent[i - 1][5]
        features.update({'-1:word': word1, '-1:postag': postag1, '-1:vox1': vox1, '-1:gen1': gen1, '-1:num': num1,
            '-1:per': per1, })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        vox1 = sent[i + 1][2]
        gen1 = sent[i + 1][3]
        num1 = sent[i + 1][4]
        per1 = sent[i + 1][5]
        features.update({'+1:word': word1, '+1:postag': postag1, '+1:vox1': vox1, '+1:gen1': gen1, '+1:num': num1,
            '+1:per': per1, })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for word, postag, vox, gen, num, per, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X_tr = [sent2features(s) for s in TRsentences]
y_tr = [sent2labels(s) for s in TRsentences]

X_te = [sent2features(s) for s in TEsentences]
y_te = [sent2labels(s) for s in TEsentences]

print("Train", len(y_tr))
print("T_test", len(y_te))

# Classifier training...

print("CRF algorithm")

start_train = datetime.datetime.now()
print(start_train)

crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100000, all_possible_transitions=False)
crf.fit(X_tr, y_tr)
end_train = datetime.datetime.now()
print(end_train)
print('Train time : {}'.format(end_train - start_train))

labels = list(crf.classes_)
labels.remove('O')
labels

start_test = datetime.datetime.now()
print(start_test)

# Make prediction...

Predict = cross_val_predict(estimator=crf, X=X_te, y=y_te, cv=5)
Predict = crf.predict(X_te)

end_test = datetime.datetime.now()
print(end_test)
print('Test time : {}'.format(end_test - start_test))

# Classifier evaluation...

report = flat_classification_report(y_te, Predict, labels=labels)
print(report)

# Compute accuracy...

a = sequence_accuracy_score(y_te, Predict)
print(a)

end_time = datetime.datetime.now()
print(end_time)
print('Execution Time (Overall Time): {}'.format(end_time - start_time))
