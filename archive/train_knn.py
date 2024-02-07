import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from wordcut_utils import process_text
from scipy.stats import randint

def train_and_save_model(file_path='./data/twitter_training.csv', model_filename='knn_model.joblib'):
    # Load data
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna(subset=['log_error_text', 'sentiment', 'log_name'])

    df['processed_text'] = df['log_error_text'].apply(process_text)

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

    # Build the KNN model
    knn_model = KNeighborsClassifier()

    # Define the hyperparameter grid
    param_dist = {
        'n_neighbors': randint(1, 20),
    }

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(knn_model, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=1, scoring='accuracy', random_state=42)

    # Fit the model with RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Save the best KNN model
    dump((best_model, text_vectorizer, name_vectorizer), model_filename)
    print(f"Best KNN model saved as {model_filename}")

    # Predictions on the test set using the best KNN model
    predictions = best_model.predict(X_test)

    # Evaluate the best KNN model
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Best KNN Model Evaluation:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    train_and_save_model()
