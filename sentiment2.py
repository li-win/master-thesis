# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:22:55 2019

@author: Lisa
"""
import nltk
from nltk.corpus import stopwords
import codecs
import spacy
import pandas as pd
import xlsxwriter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob_de import TextBlobDE as TextBlob

nlp = spacy.load('de_core_news_sm')
stop = stopwords.words('german')
stop_words = ['aber', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'er', 'ihn', 'ihm', 'ein', 'eine', 'einer', 'einem', 'einen', 'eines', 'es', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'indem', 'ins', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'noch', 'nun', 'nur', 'ob', 'oder', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unsere', 'unserem', 'unseren', 'unser', 'unseres', 'unter', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 'zur', 'zwar', 'zwischen']
#print("Stoppworte: \n", stop)

#dialekt = {"'n": " ein", "'ne": " eine", "'nem": " einem", "'ner": " einer", "'nen": "einen"}
dialekt2 ={"'n": "en", "'res": "eres", "'nem": " einem"}

file = open("Achtundneunzig Luftballons (98 Luftballons).txt", "r", encoding="utf-8")
text = file.read()
file.close()

doc = nlp(text)
param = [[token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.is_alpha, token.is_stop] for token in nlp(text)]
df = pd.DataFrame(param)
headers = ['text', 'lemma', 'pos', 'tag', 'dep', 'is_alpha', 'is_stop']
df.columns = headers

#writer = pd.ExcelWriter('prep_text.xlsx', engine='xlsxwriter')
#df.to_excel(writer, sheet_name='Tabelle1')
#writer.save()

tokens = [token.text for token in doc]

#Lemma-Liste ohne Stoppworte, Leerzeichen oder Punktionszeichen
lemmas = [token.lemma_ for token in doc if token.pos_ != 'SPACE' and token.pos_ != 'PUNCT' and token.lemma_ not in stop_words]
        
print("\n Lemma-Liste: \n", lemmas)

lemmas_verb = [token.lemma_ for token in doc if token.pos_ == 'VERB']

#Dialekt umwandeln in Hochdeutsch (geh'n -> gehen)
new_sent =[]
    
for token in tokens:
    for replacement in dialekt2:
        if replacement in token:
            token = token.replace(replacement, dialekt2[replacement])
    new_sent.append(token)
    
lemmas = " ".join(new_sent)
print("\n neue Lemma-Liste: \n", lemmas)

doc2 = nlp(lemmas)
#NER
#for ent in doc2.ents:
    #print(ent.text, ent.start_char, ent.end_char, ent.label_)

blob = TextBlob(lemmas)
#print(blob.tags)
#print(blob.noun_phrases)

result2 = []
for sentence in blob.sentences:
    panda = sentence.sentiment.polarity
    result2.append(panda)
    
    print("\n Sentiment pro Satz: \n",sentence, panda)

print("\n Sentiment Durchschnitt Gesamt: ")
print(sum(result2)/len(result2))


"""
sentences = nltk.sent_tokenize(text, language='german')

tokenized_text = [nltk.word_tokenize(sent, language='german') for sent in sentences]

#prep_text = [word for word in tokenized_text if not word in stop]
prep_text = []
for word in tokenized_text:
    if word not in stop:
        prep_text.append(word)

#print(tokenized_text)
print(prep_text)
"""
#Text: The original word text.
#Lemma: The base form of the word.
#POS: The simple part-of-speech tag.
#Tag: The detailed part-of-speech tag.
#Dep: Syntactic dependency, i.e. the relation between tokens.
#Shape: The word shape – capitalisation, punctuation, digits.
#is alpha: Is the token an alpha character?
#is stop: Is the token part of a stop list