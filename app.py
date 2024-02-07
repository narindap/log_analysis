import pandas as pd
from unsupervised import similar_word, train_fasttext_model
from train_logisticregression import train_and_save_model
from predict_model import predict_sentiment

def train_fasttext():
    # Train fasttext
    category_unsup = './data/category.csv'
    model_locate = './fasttext_model.bin'
    model_settings = {
        'model': 'skipgram',
        'minCount': 1,
        'epoch': 600,
        'word_ngrams': 1,
        'dim': 300
    }
    train_fasttext_model(category_unsup, model_settings, model_locate)

def train_LR():
    # Train LR
    LR_model = './LR_model.joblib'
    file_LR = './data/filter_twitter_training.csv'
    rf_params = {'n_estimators': 150, 'random_state': 42}
    train_and_save_model(file_LR, LR_model)

def predict_fasttext(input_text, model_locate):
    # Predict using fasttext
    return similar_word(input_text, model_locate)

def predict_LR(test_text, tmpword, model_filename):
    # Predict using LR
    test_names = 'SomeName'
    predict_sentiment([test_text], [test_names], model_filename)

if __name__ == "__main__":
    # Comment out the training if it's already done
    # train_fasttext()
    train_LR()

    # Read the CSV file
    # csv_file_path = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/test.csv'
    # df = pd.read_csv(csv_file_path)

    # # Iterate through each row and perform predictions
    # for index, row in df.iterrows():
    #     input_text = row['log_error_text']
    #     model_locate = "fasttext_model.bin"
        
    #     # Perform prediction using fasttext
    #     tmpword = predict_fasttext(input_text, model_locate)
        
    #     # Print the result
    #     print(f"Prediction for log_id {row['log_id']}: {tmpword}")

    #     # Optionally, you can also perform LR prediction here if needed
    #     model_filename = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/LR_model.joblib'
    #     predict_LR(input_text, tmpword, model_filename)


    # # Predict using fasttext
    # input_text = "@ amazon I think this is in bad taste and a terrible exploitation of a man's death. they claim BLM, but here we are. pic.twitter.com "
    # model_locate = "fasttext_model.bin"
    # tmpword = predict_fasttext(input_text, model_locate)
    # print(tmpword)

    # # Predict using LR
    # model_filename = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/LR_model.joblib'
    # predict_LR(input_text, tmpword, model_filename
