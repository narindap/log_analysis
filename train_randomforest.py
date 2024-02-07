import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from wordcut_utils import preprocess_dataset
from scipy.stats import randint

def train_and_save_model(file_path='./data/twitter_training.csv', 
                         model_filename='./model/random_forest_model.joblib'):

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['topic', 'sentiment', 'text'])
    df['data'] = df['text'].apply(preprocess_dataset)

    text_vectorizer = CountVectorizer()
    X_text = text_vectorizer.fit_transform(df['text'])

    topic_vectorizer = CountVectorizer()
    X_name = topic_vectorizer.fit_transform(df['topic'])

    X_combined = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_name.toarray())], axis=1)
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()

    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4)
    }

    #RandomizedSearchCV
    random_search = RandomizedSearchCV(rf_model, 
                                       param_distributions=param_dist, 
                                       n_iter=10, 
                                       cv=5, 
                                       n_jobs=-1, 
                                       verbose=1, 
                                       scoring='accuracy', 
                                       random_state=42)

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    dump((best_model, text_vectorizer, topic_vectorizer), model_filename)
    print(f"Best model saved as {model_filename}")

    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Best Model Evaluation:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    train_and_save_model()
