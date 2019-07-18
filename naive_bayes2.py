# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:50:39 2019
Python 3.7
@author: Lisa
"""
#Naive Bayes ohne vorherige Entfernung von Stopworten

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.DataFrame()
df = pd.read_excel('sätze mit sentiment 3.xlsx', encoding='utf-8')
print(df.head(3))

df['Label'] = 0
df.loc[df['Sentiment'] > 0.2, 'Label'] = 1
df.loc[df['Sentiment'] < -0.2, 'Label'] = -1

df.to_excel('satz_sent_label2.xlsx', index=False, header=True)

#Wieviel positive bzw. negative Sätze im dataset
print("\n ***Value Counts*** \n")
print(df.Label.value_counts())
print(df.Label.value_counts(normalize=True) * 100)

x = df.Sätze
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vect = CountVectorizer(max_features=1000, binary=True)

x_train_vect = vect.fit_transform(x_train)

sm = SMOTE()

x_train_res, y_train_res = sm.fit_sample(x_train_vect, y_train)

unique, counts = np.unique(y_train_res, return_counts=True)
print("\n ausbalanciertes Trainings-Set \n")
print(list(zip(unique, counts)))

nb = MultinomialNB()

nb.fit(x_train_res, y_train_res)

print(nb.score(x_train_res, y_train_res))

x_test_vect = vect.transform(x_test)

y_pred = nb.predict(x_test_vect)

print(y_pred)

print("\n Genauigkeit:", accuracy_score(y_test, y_pred))
print("\n F1-Score:", f1_score(y_test, y_pred, average='macro'))
print("\n Konfusionsmatrix: \n", confusion_matrix(y_test, y_pred))
