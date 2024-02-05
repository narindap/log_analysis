import fasttext
import tempfile

import pandas as pd
from wordcut_utils import wordcut_input, wordcut_unsupervised
from predict_model import predict_sentiment

df = pd.read_csv('./data/category.csv')
log_data = df['log_error']  # Assuming 'log_error' is the column you want to use

def train_fasttext_model(log_data):
    processed_text = wordcut_unsupervised(log_data)
    print(processed_text)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(processed_text)

    model = fasttext.train_unsupervised(temp_file.name, **model_settings)

    return model

def similar_word(input_text, log_data):
    processed_text = wordcut_input([input_text])
    print(processed_text)
    words_per_line = 2
    if len(processed_text) % 2 != 0:
        words_per_line = 3

    model = train_fasttext_model(log_data)
    words = processed_text.split()
    chunks = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]

    max_similarity = -1
    max_chunk = None
    max_word = None

    for chunk in chunks:
        chunk_sentence = ' '.join(chunk)
        similar_words = model.get_nearest_neighbors(chunk_sentence, k=1)

        if similar_words:
            similarity, word = similar_words[0]  # Take the first (most similar) word
            if similarity > max_similarity:
                max_similarity = similarity
                max_chunk = chunk_sentence
                max_word = word

    if max_chunk is not None:
        print(f"[ {max_chunk} ] \n########## {max_similarity} : {max_word} ##########")
        predict_sentiment([input_text], [max_word])

if __name__ == "__main__":
    model_settings = {
        'model': 'skipgram',
        'minCount': 1,
        'epoch': 600,
        'word_ngrams': 1,
        'dim': 300
    }
    print("Enter your sentence:")
    text_input = input()
    similar_word(text_input, log_data)
