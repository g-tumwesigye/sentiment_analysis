# Sentiment Analysis on IMDB Movie Reviews

This project focuses on **Sentiment Analysis** using the IMDB 50K movie reviews dataset. The goal is to build a **text classification system** that can accurately determine whether a movie review is **positive** or **negative**, comparing traditional machine learning and deep learning models.

---

## Dataset

- **Source**: [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Format**: CSV
- **Columns**:
  - `review`: The full text of the movie review.
  - `sentiment`: The sentiment label (`positive` or `negative`).

---

## Team Members

| Name                     | Role / Contribution                                  |
|--------------------------|------------------------------------------------------|
| **John Deng Madit Akech*   | Data loading, cleaning, EDA, word count, handoff     |
| **Makuach Teny Gatluak**| Traditional ML model (Naïve Bayes/SVM + TF-IDF)      |
| **Geofrey Tumwesigye** | Deep Learning model (LSTM or GRU + Word Embeddings)  |
| **Peris Nyawira Wangui* | Experiment tables, results discussion, report writing|

---

## Preprocessing

- Lowercased all text
- Removed HTML tags and punctuation
- Removed extra whitespaces
- Added word count feature for analysis
- Tokenization and stopword removal handled during vectorization

---

## Exploratory Data Analysis (EDA)

- Sentiment distribution (positive vs negative)
- Histogram of word counts in reviews
- Boxplot of word count by sentiment

---

## Models Implemented

1. **Traditional Machine Learning**
   - Naïve Bayes / SVM
   - TF-IDF Vectorization

2. **Deep Learning**
   - LSTM / GRU
   - Word Embeddings (GloVe / Word2Vec)

---

## Experiments

Two experiment tables are included in the report comparing:
- Learning rate variations
- Batch sizes
- Optimizers
- Feature extraction methods (TF-IDF vs GloVe)

---

## Evaluation Metrics

- Accuracy
- F1-Score
- Precision
- Confusion Matrix
- Cross-Entropy Loss (for deep models)

---

## How to Run

### Requirements
- Python ≥ 3.7
- Google Colab (recommended)
- TensorFlow, Scikit-learn, NLTK, Seaborn, Matplotlib, Pandas

### Instructions

1. Open the project in Google Colab.
2. Upload the original `IMDB Dataset.csv` 
3. Run each section sequentially:
   - Preprocessing
   - EDA
   - Model training
   - Evaluation
4. Review experiment results and graphs.

---

## Report

A full PDF report is included in this repository. It contains:
- Dataset explanation
- Preprocessing steps
- Model architecture
- Experiment tables
- Results discussion
- Team contributions

---

## Github Link

- [GitHub Repository URL](https://github.com/g-tumwesigye/sentiment_analysis)

---

## 📚 References

- [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


