import pandas as pd
from unsupervised import similar_word, train_fasttext_model

from predict_model import predict_sentiment
from train_randomforest import train_and_save_model
from scipy.stats import randint

def train_fasttext():
    # Train FastText
    file_path = pd.read_csv('./data/category.csv')
    data_category = file_path['keyword']
    model_path = './model/fasttext_model.bin'
    model_settings = {
        'model': 'skipgram',
        'minCount': 1,
        'epoch': 600,
        'word_ngrams': 1,
        'dim': 300
    }
    train_fasttext_model(data_category, model_settings, model_path)


def train_RF():
    
    model_path ='./model/random_forest_model.joblib'
    file_path = './data/twitter_training.csv'
    model_setting = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
    }

    train_and_save_model(file_path, model_path,model_setting)

def predict_fasttext(input_text):
    # Predict using fasttext
    # input_text = "@ amazon I think this is in bad taste and a terrible exploitation of a man's death. they claim BLM, but here we are. pic.twitter.com "
    model_path = "./model/fasttext_model.bin"
    tmpword = similar_word(input_text, model_path)
    return tmpword

def predict_RF(input_text, tmpword):
    model_path = './model/random_forest_model.joblib'
    predict_sentiment([input_text], [tmpword], model_path)



if __name__ == "__main__":
    # Comment out the training if it's already done
    # train_fasttext()
    # train_RF()


    print("input text")
    input_text = input()

    # Predict using fasttext
    tmpword = predict_fasttext(input_text)
    print(tmpword)
    predict_RF(input_text,tmpword)

    