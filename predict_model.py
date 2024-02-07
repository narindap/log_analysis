import pandas as pd
from joblib import load

def predict_sentiment(test_texts, test_topics, model='./model/random_forest_model.joblib'):
    # Load the trained model and vectorizers
    loaded_model, text_vectorizer, name_vectorizer = load(model)

    # Feature extraction for additional data
    X_test_text = text_vectorizer.transform(test_texts)
    X_test_topic = name_vectorizer.transform(test_topics)

    X_test_combined = pd.concat([pd.DataFrame(X_test_text.toarray()), pd.DataFrame(X_test_topic.toarray())], axis=1)

    # Make predictions
    predictions_additional = loaded_model.predict(X_test_combined)

    # Display predicted sentiment for additional data
    print("Predicted Sentiment for Additional Data:", predictions_additional)

    # Display predicted probabilities for additional data
    predicted_proba = loaded_model.predict_proba(X_test_combined)
    print("Predicted Probabilities for Additional Data:")
    for i, class_name in enumerate(loaded_model.classes_):
        print(f"Probability of {class_name}: {predicted_proba[0][i]}")