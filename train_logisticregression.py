import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

def train_and_save_model(file_path='./data/twitter_training.csv', model_filename='logistic_regression_model.joblib'):
    # Load data
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna(subset=['log_error_text', 'sentiment', 'log_name'])

    # Feature extraction
    text_vectorizer = CountVectorizer()
    X_text = text_vectorizer.fit_transform(df['log_error_text'])

    name_vectorizer = CountVectorizer()
    X_name = name_vectorizer.fit_transform(df['log_name'])

    X_combined = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_name.toarray())], axis=1)

    # Target variable
    y = df['sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model
    dump((model, text_vectorizer, name_vectorizer), model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
