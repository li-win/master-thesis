# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:52:31 2019
Python 3.7
@author: Lisa
"""

import os,glob
import openpyxl
import pandas as pd
import spacy
from textblob_de import TextBlobDE as TextBlob
import nltk
import regex as re
import itertools

nlp = spacy.load('de_core_news_sm')
stop_words = ['aber', 'als', 'also', 'am', 'an', 'ander', 'andere', 
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
dialekt2 ={"'n": "en", "'res": "eres", "'nem": " einem"}

wb = openpyxl.Workbook()
sheet = wb.active
index = 0
texts = []

folder_path = 'E:\lisa-\OneDrive\Giessen\Masterarbeit\korpus'
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        texts.append(text)

print(len(texts))

df = pd.DataFrame({"Text":texts})
#print(df)

i = 0
text_sent_list =[]
satz_sent_list = []
sentences2 = []
#338
while i<338:
    prep = texts[i]
        
    doc = nlp(prep)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc if token.pos_ != 'SPACE' and token.pos_ != 'PUNCT']
    
    #Dialekt umwandeln in Hochdeutsch (geh'n -> gehen)
    new_sent =[]
        
    for token in tokens:
        for replacement in dialekt2:
            if replacement in token:
                token = token.replace(replacement, dialekt2[replacement])
        new_sent.append(token)
        
    lemmas = " ".join(new_sent)
    
    #strings in Liste mit tokenisierten Sätzen (ein Satz ist ein Listenelement) umwandeln
    #lemmas = re.sub(r'[^\w\s]','',lemmas) #Alles entfernen außer Worte und Leerzeichen
    sentences = nltk.sent_tokenize (lemmas, language ="german")
    tokenized_text = [nltk.word_tokenize(sent, language="german") for sent in sentences]
    #print(tokenized_text)
    for sent in sentences:
        sentences2.append(sent)
        
    result=[]
    blob = TextBlob(lemmas)
    for sentence in blob.sentences:
        var1 = sentence.sentiment.polarity
        result.append(var1)
        #print(result)
        
    #jeden Sentimentwert eines Satzes in Liste abspeichern    
    satz_sent_list.append(result)
    text_sent = sum(result)/len(result)
    
    #print(text_sent)
    text_sent_list.append(text_sent)
       
    i +=1

print(len(satz_sent_list)) 

#Liste aus Listen zu einer Liste umformen
new_satz_sent_list = list(itertools.chain.from_iterable(satz_sent_list))

#print(satz_sent_list)
df2 = pd.DataFrame({"Sätze":sentences2})
df3 = pd.DataFrame(new_satz_sent_list)
df2['Sentiment'] = new_satz_sent_list
export_excel = df2.to_excel("sätze3.xlsx", index=False, header=True)
export_excel = df3.to_excel("please3.xlsx", index=False, header=True)
