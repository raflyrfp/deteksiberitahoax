import pandas as pd # type: ignore
import string
import joblib # type: ignore
import os

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report # type: ignore

# Buat folder 'model' jika belum ada
os.makedirs('model', exist_ok=True)

# Fungsi pembersih teks
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load dataset
df = pd.read_csv('Data_Hoaks_2023.csv', encoding='latin1', delimiter=';')

# Hapus kolom yang tidak perlu
df = df.loc[:, ['Konten', 'Label']].dropna()

# Bersihkan teks
df['clean_text'] = df['Konten'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluasi
print(classification_report(y_test, model.predict(X_test)))

# Simpan model dan vectorizer
joblib.dump(model, 'model/logistic_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

print("✅ Model dan vectorizer berhasil disimpan di folder 'model/'")