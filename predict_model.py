import pandas as pd
from joblib import load

def predict_sentiment(test_error_texts, test_names, model_filename):
    # Load the trained model and vectorizers
    loaded_model, text_vectorizer, name_vectorizer = load(model_filename)

    # Feature extraction
    X_test_text = text_vectorizer.transform(test_error_texts)
    X_test_name = name_vectorizer.transform(test_names)

    X_test_combined = pd.concat([pd.DataFrame(X_test_text.toarray()), pd.DataFrame(X_test_name.toarray())], axis=1)

    # prediction
    predictions_additional = loaded_model.predict(X_test_combined)

    # Display predicted sentiment
    print("Predicted Sentiment for Additional Data:", predictions_additional)

    # Display predicted probabilities
    predicted_proba = loaded_model.predict_proba(X_test_combined)
    print("Predicted Probabilities for Additional Data:")
    for i, class_name in enumerate(loaded_model.classes_):
        print(f"Probability of {class_name}: {predicted_proba[0][i]}")