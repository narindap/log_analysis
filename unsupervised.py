import fasttext
import tempfile
from wordcut_utils import wordcut, wordcut_model

category = './data/category.csv'
#log_input = './data/input.txt'


def train_fasttext_model(log_data):
    with open(log_data, 'r', encoding='utf-8') as log_file:
        processed_text = wordcut_model(log_file)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(processed_text)

    model = fasttext.train_unsupervised(temp_file.name,**model_settings)

    return model

def similar_word(input_text, log_data):
    #with open(input_file_path, 'r', encoding='utf-8') as input_file:
        #processed_text = wordcut(input_file)
    processed_text = wordcut([input_text])
    words_per_line = 1
    model = train_fasttext_model(log_data)
    words = processed_text.split()
    chunks = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]
    for chunk in chunks:
        chunk_sentence = ' '.join(chunk)
        similar_words = model.get_nearest_neighbors(chunk_sentence, k=3)
        for word, similarity in similar_words:
            if float(word) > 0.5:
                print(f"[ {chunk_sentence} ] \n########## {similarity} : {word} ##########")


if __name__ == "__main__":

    model_settings = {
    'model': 'skipgram',
    'minCount': 1,
    'epoch': 450,
    'word_ngrams': 1,
    'dim': 100
}
    print("Enter your sentence:")
    text_input = input()
    similar_word(text_input, category)
