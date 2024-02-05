import re
import spacy
from nltk import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def wordcut_input(input_text):
    sentences = []

    for line in input_text:
        line = re.sub(r'[^a-zA-Z0-9 ]', '', line).strip()
        doc = nlp(line)
        english_words = [stemmer.stem(token.lemma_.lower()) for token in doc]
        sentence = ' '.join(english_words)
        sentences.append(sentence)
    return ''.join(sentences)
    
def wordcut_unsupervised(input_text):
    sentences = []

    for line in input_text:
        line = re.sub(r'[^a-zA-Z0-9 ]', '', line).strip()
        doc = nlp(line)
        english_words = [stemmer.stem(token.lemma_.lower()) for token in doc]
        sentence = ''.join(english_words)
        sentences.append(sentence)
    print(sentence)
    return ' '.join(sentences)

def process_text(text):
    doc = nlp(text)
    stemmed_and_lemmatized = [stemmer.stem(token.lemma_.lower()) for token in doc]
    return ' '.join(stemmed_and_lemmatized)