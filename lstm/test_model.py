import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loaded_model = load_model('/Users/nakarin.rue/Documents/log_analyze/log_analysis/model_supervised.h5')

# Load new data from a CSV file (replace 'new_data.csv' with your file path)
new_data = pd.read_csv('/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/predict_twitter.csv')

# Handle missing values by replacing NaN with an empty string
new_data['PF'] = new_data['PF'].fillna('')
new_data['SEN'] = new_data['SEN'].fillna('')

# Convert text to lowercase
new_data['PF'] = new_data['PF'].str.lower()
new_data['SEN'] = new_data['SEN'].str.lower()

# Combine 'PF' and 'SEN' text
combined_text = new_data['PF'] + ' ' + new_data['SEN']

# Preprocess the new data using the same Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_text)
X_new = tokenizer.texts_to_sequences(combined_text)
X_new = pad_sequences(X_new, maxlen=108)  # Make sure to pad sequences with the same length as the training data

# Make predictions
predictions = loaded_model.predict(X_new)

# Decode predictions if needed (e.g., for classification)
predicted_classes = predictions.argmax(axis=-1)

# Print the predicted classes
print("Predicted Classes:", predicted_classes)
