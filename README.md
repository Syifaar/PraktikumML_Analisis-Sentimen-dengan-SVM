# 📊 Laporan Latihan Sentiment Analysis with SVM (Support Vector Machine)

## 📝 Deskripsi
Proyek ini merupakan implementasi analisis sentimen terhadap review teks menggunakan algoritma **Support Vector Machine (SVM)**. Dataset yang digunakan merupakan kumpulan review yang diklasifikasikan ke dalam sentimen positif (1) dan negatif (0).

---
## 📂 Dataset
Dataset diunduh dari: [SI650 Winter 2011 - Kaggle](https://www.kaggle.com/c/si650winter11/data)  
- Jumlah data: **6931 review**
---

## 🔄 Preprocessing
- **Tokenisasi** menggunakan `TextBlob`
- **Lemmatization**: Mengubah kata ke bentuk dasar menggunakan `.lemma`
- **Vectorization**:
  - **CountVectorizer** untuk menghitung frekuensi kata
  - **TF-IDF Transformer** untuk menormalisasi dan memberikan bobot pada kata penting

---

## ⚙️ Model: SVM (Support Vector Machine)
Model dibangun dalam sebuah pipeline:

```python
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=to_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),
])

```
Model diuji dengan menggunakan GridSearchCV dan StratifiedKFold Cross-Validation.

## 🎯 Hyperparameter yang diuji:
Kernel: linear, rbf
C: 1, 10, 100, 1000
Gamma (untuk rbf): 0.001, 0.0001

## Hasil Evaluasi
Akurasi: 99%
Metode terbaik: SVM dengan kernel RBF, C=1000, gamma=0.001

## 🧪 Contoh Prediksi
```
classifier.predict(["the vinci code is awesome"])  # Output: 1 (positif)
classifier.predict(["the vinci code is bad"])      # Output: 0 (negatif)
```

## 📌 Kesimpulan
Model SVM terbukti sangat efektif dalam mengklasifikasi sentimen review teks. Dengan preprocessing yang tepat dan pemilihan parameter terbaik, model mampu mencapai akurasi sangat tinggi.

