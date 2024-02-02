import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# อ่านข้อมูลจากไฟล์ CSV
file_path = './data/twitter_training.csv'  # ระบุที่อยู่ของไฟล์ CSV
df = pd.read_csv(file_path)

# ตรวจสอบและข้ามข้อมูลที่มีค่าว่าง
df = df.dropna(subset=['log_error_text', 'sentiment', 'log_name'])

# แสดงข้อมูลใน DataFrame
print("Cleaned DataFrame:")
print(df)

# แปลงข้อมูล 'log_error_text' เป็นเวกเตอร์
text_vectorizer = CountVectorizer()
X_text = text_vectorizer.fit_transform(df['log_error_text'])

# แปลงข้อมูล 'log_name' เป็นเวกเตอร์
name_vectorizer = CountVectorizer()
X_name = name_vectorizer.fit_transform(df['log_name'])

# รวมเวกเตอร์ทั้งสอง
X_combined = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_name.toarray())], axis=1)

# แปลง label 'sentiment' เป็นตัวเลข
y = df['sentiment']

# แบ่งข้อมูลเป็นชุดฝึกฝนและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ทดสอบโมเดล
predictions = model.predict(X_test)

# ประเมินผลลัพธ์
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# ข้อมูลทดสอบเพิ่มเติม
test_log_error_texts = ['Looting loot while having my loot looted. Borderlands 3.. . twitch.tv/teegan']
test_log_names = ['Borderlands']

# แปลงข้อมูลทดสอบเพิ่มเติมเป็นเวกเตอร์
X_test_text = text_vectorizer.transform(test_log_error_texts)
X_test_name = name_vectorizer.transform(test_log_names)

# รวมเวกเตอร์ทั้งสอง
X_test_combined = pd.concat([pd.DataFrame(X_test_text.toarray()), pd.DataFrame(X_test_name.toarray())], axis=1)

# ทำนาย
predictions_additional = model.predict(X_test_combined)

# แสดงผลลัพธ์ที่ทำนายได้
print("Predicted Sentiment for Additional Data:", predictions_additional)

predicted_proba = model.predict_proba(X_test_combined)
print("Predicted Probabilities for Additional Data:")
for i, class_name in enumerate(model.classes_):
    print(f"Probability of {class_name}: {predicted_proba[0][i]}")