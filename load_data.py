from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import pandas as pd
import numpy as np

sTrain = "dataCsv/sentence_train.csv"
sTest = "dataCsv/sentence_test.csv"
lTrain = "dataCsv/label_train.csv"
lTest = "dataCsv/label_test.csv"
df1 = pd.read_csv(sTrain, sep=',')
df2 = pd.read_csv(sTest, sep=',')
df3 = pd.read_csv(lTrain, sep=',')
df4 = pd.read_csv(lTest, sep=',')

sentences_train = df1['0'].values
sentences_test = df2['0'].values
y_train = df3['0'].values
y_test = df4['0'].values


def load_imdb_sentiment_analysis_dataset(sTrain, sTest, lTrain, lTest, seed=123):
    train_texts = sTrain
    train_labels = lTrain
    test_texts = sTest
    test_labels = lTest
    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))
