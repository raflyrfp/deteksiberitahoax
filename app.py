from flask import Flask, request, render_template # type: ignore
import joblib # type: ignore

# Inisialisasi Flask app
app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load('model/logistic_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        teks = request.form['berita']
        vectorized_text = vectorizer.transform([teks])
        prediction = model.predict(vectorized_text)

        hasil = 'Hoaks' if prediction[0] == 1 else 'Valid'
        return render_template('index.html', prediction_text=f"Hasil Deteksi: {hasil}")

if __name__ == '__main__':
    app.run(debug=True)
