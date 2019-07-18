"""Demonstrates how to make a simple call to the Natural Language API."""

import argparse
import os

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

#authentication
credential_path = "E:\lisa-\OneDrive\Giessen\Masterarbeit\My Project-eb1d60dd1658.json"
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'filename',
        help='The filename of the text you\'d like to analyze.')
    args = parser.parse_args()

    analyze(args.filename)
