import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load your dataset
df = pd.read_csv('/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/filter_twitter_training.csv')

# Handle missing values by replacing NaN with an empty string
df['PF'] = df['PF'].fillna('')
df['SEN'] = df['SEN'].fillna('')

# Convert text to lowercase
df['PF'] = df['PF'].str.lower()
df['SEN'] = df['SEN'].str.lower()

# Combine 'PF' and 'SEN' text
combined_text = df['PF'] + ' ' + df['SEN']

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_text)  # Tokenize lowercase combined text
X = tokenizer.texts_to_sequences(combined_text)
X = pad_sequences(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['ST'])
y_categorical = to_categorical(y)  # Convert to one-hot encoding for multi-class

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(LSTM(100, dropout=0.65))
optimizer = Adam(learning_rate=0.01)  # Set your desired learning rate
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_test, y_test))

# Assuming your model is already trained and stored in the 'model' variable
model.save('./model_supervised.h5')
