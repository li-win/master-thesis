"""simple call to the Natural Language API."""

import argparse
import os

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

#authentication
credential_path = "..."
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

sentiment_list = []

def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))
        sentiment_list.append(sentence_sentiment)

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    with open('1_Ausgabe_sentiments.txt', 'a') as f:
        print(sentiment_list, file=f)
    with open('2_Ausgabe_scores.txt', 'a') as f:
        print(score, file=f)
    with open('3_Ausgabe_magnitudes.txt', 'a') as f:
        print(magnitude, file=f)
    return 0

def analyze(filename):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(filename, 'r') as file:
        # Instantiates a plain text document.
        content = file.read()

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)

    # Print the results
    print_result(annotations)

i=0
while i < 338:
    filenames = os.listdir('E:\lisa-\OneDrive\Giessen\Masterarbeit\korpus2')
    analyze(filenames[i])
    i += 1
