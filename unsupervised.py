import fasttext
import tempfile
import pandas as pd
from wordcut_utils import preprocess_input, preprocess_unsupervised
from predict_model import predict_sentiment


df = pd.read_csv('./data/category.csv')
data_category = df['keyword']

# Function to train FastText model
def train_fasttext_model(data_category,model_settings,model_path):
    # Preprocess text
    processed_text = preprocess_unsupervised(data_category)
    
    # Write processed text 
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(processed_text)

    # Train FastText model 
    model = fasttext.train_unsupervised(temp_file.name, **model_settings)
    model.save_model(model_path)

    return model

# Function to find a similar word
def similar_word(input_text, model_path):
    # Preprocess input text using wordcut
    processed_text = preprocess_input([input_text])

    # Determine the number of words per line for FastText
    words_per_line = 2
    if len(processed_text) % 2 != 0:
        words_per_line = 3

    # Train FastText
    model_fasttext = fasttext.load_model(model_path)

    # Split input into chunks
    words = processed_text.split()
    chunks = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]

    # Find the most similar word for each chunk
    max_similarity = -1
    max_chunk = None
    max_word = None

    for chunk in chunks:
        chunk_sentence = ' '.join(chunk)
        similar_words = model_fasttext.get_nearest_neighbors(chunk_sentence, k=1)

        if similar_words:
            similarity, word = similar_words[0]  # Take the first (most similar) word
            if similarity > max_similarity:
                max_similarity = similarity
                max_chunk = chunk_sentence
                max_word = word

    # Print the result
    if max_chunk is not None:
        print(f"[ {max_chunk} ] \n########## {max_similarity} : {max_word} ##########")
        # Predict sentiment using the input text and the most similar word
        # predict_sentiment([input_text], [max_word])
    return similar_word

# Main section
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

    similar_word(text_input, data_category)
