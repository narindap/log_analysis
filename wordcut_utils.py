import re
import spacy
from nltk import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def clean_data(line):
    line = re.sub(r'[^a-zA-Z0-9 ]', '', line).strip()
    doc = nlp(line)
    english_words = [stemmer.stem(token.lemma_.lower()) for token in doc]
    return english_words

def preprocess_input(input_text):
    preprocessed_sentences = []

    for line in input_text:
        english_words = clean_data(line)
        sentence = ' '.join(english_words)
        preprocessed_sentences.append(sentence)
    return ''.join(preprocessed_sentences)
    
def preprocess_unsupervised(input_text):
    preprocessed_sentences = []

    for line in input_text:
        english_words = clean_data(line)
        sentence = ''.join(english_words)
        preprocessed_sentences.append(sentence)
    return ' '.join(preprocessed_sentences)

def preprocess_dataset(text):

    doc = nlp(text)
    stemmed_and_lemmatized = [stemmer.stem(token.lemma_.lower()) for token in doc]
    return ' '.join(stemmed_and_lemmatized)
