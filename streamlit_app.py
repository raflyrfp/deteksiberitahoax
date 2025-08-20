import joblib

# Load model dan vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model/logistic_model.pkl")       # sama persis dengan path di Flask
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Judul aplikasi
st.title("📰 Deteksi Berita Hoaks")
st.write("Masukkan teks berita untuk dideteksi apakah **Hoaks** atau **Valid**.")

# Input berita
teks = st.text_area("Masukkan teks berita:")

# Tombol prediksi
if st.button("Deteksi"):
    if teks.strip() == "":
        st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
    else:
        vectorized_text = vectorizer.transform([teks])
        prediction = model.predict(vectorized_text)

        hasil = "🚨 Hoaks" if prediction[0] == 1 else "✅ Valid"
        st.subheader(f"Hasil Deteksi: {hasil}")