# 📊 Sentiment Analysis with SVM (Support Vector Machine)

## 📝 Deskripsi
Proyek ini merupakan implementasi analisis sentimen terhadap review teks menggunakan algoritma **Support Vector Machine (SVM)**. Dataset yang digunakan merupakan kumpulan review yang diklasifikasikan ke dalam sentimen positif (1) dan negatif (0).

---

## 📂 Dataset
Dataset diunduh dari: [SI650 Winter 2011 - Kaggle](https://www.kaggle.com/c/si650winter11/data)  
- Format: `.txt` (Tab-Separated)
- Jumlah data: **6931 review**
- Kolom:
  - `liked`: Label (0 = Negatif, 1 = Positif)
  - `text`: Review teks

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
