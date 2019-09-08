# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:16:13 2019
Python 3.7
@author: Lisa Winkler
"""

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

import time
start = time.time()

stop_words2 = ['aber', 'als', 'also', 'am', 'an', 'ander', 'andere', 
              'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 
              'andern', 'anderer', 'anders', 'auch', 'auf', 'bei', 
              'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 
              'den', 'des', 'dem', 'die', 'das', 'daß', 'derselbe', 
              'derselben', 'denselben', 'desselben', 'demselben', 
              'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 
              'deine', 'deinem', 'deinen', 'deiner', 'deines', 
              'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 
              'dies', 'diese', 'diesem', 'diesen', 'dieser', 
              'dieses', 'doch', 'dort', 'durch', 'er', 'ihn', 'ihm', 
              'ein', 'eine', 'einer', 'einem', 'einen', 'eines', 
              'es', 'euer', 'eure', 'eurem', 'euren', 'eurer', 
              'eures', 'gewesen', 'hab', 'habe', 'haben', 'hat', 
              'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 
              'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 
              'ihres', 'euch', 'im', 'indem', 'ins', 'jede', 'jedem', 
              'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 
              'jener', 'jenes', 'jetzt', 'man', 'manche', 'manchem', 
              'manchen', 'mancher', 'manches', 'mein', 'meine', 
              'meinem', 'meinen', 'meiner', 'meines', 'mit', 'nach', 
              'noch', 'nun', 'nur', 'ob', 'oder', 'sehr', 'sein', 
              'seine', 'seinem', 'seinen', 'seiner', 'seines', 
              'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 
              'solchem', 'solchen', 'solcher', 'solches', 'soll', 
              'sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 
              'unsere', 'unserem', 'unseren', 'unser', 'unseres', 
              'unter', 'vom', 'von', 'vor', 'während', 'war', 'waren', 
              'warst', 'was', 'weil', 'weiter', 'welche', 'welchem', 
              'welchen', 'welcher', 'welches', 'wenn', 'werde', 
              'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 
              'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 
              'zur', 'zwar', 'zwischen']

df = pd.DataFrame()
df = pd.read_excel('sätze3.xlsx', encoding='utf-8')
print(df.head(3))

df['Label'] = 0
df.loc[df['Sentiment'] > 0.2, 'Label'] = 1
df.loc[df['Sentiment'] < -0.2, 'Label'] = -1

#df.to_excel('satz_sent_label4.xlsx', index=False, header=True)

features = df.iloc[:, 0].values
labels = df.iloc[:, 2].values

prep_features = []

for sentence in range(0, len(features)):
    #Sonderzeichen entfernen
    prep_feature = re.sub(r'\W', ' ', str(features[sentence]))
    prep_feature = re.sub(r'^b\s+', '', prep_feature)
    #Einzelbuchstaben entfernen
    prep_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', prep_feature)
    prep_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', prep_feature)
    #überflüssige Leerzeichen entfernen
    prep_feature = re.sub(r'\s+', ' ', prep_feature, flags=re.I)
    #alles klein schreiben
    prep_feature = prep_feature.lower()
    
    prep_features.append(prep_feature)

#print(prep_features)
    
x = prep_features
y = df.Label

#min_df verringern - höhere Genauigkeit, max_df verändern - keine Veränderung
vectorizer = CountVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stop_words2)
x = vectorizer.fit_transform(x).toarray()

#Set in Trainings- und Testset splitten
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(x_train, y_train)
predictions = text_classifier.predict(x_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

end = time.time()
print(end - start)