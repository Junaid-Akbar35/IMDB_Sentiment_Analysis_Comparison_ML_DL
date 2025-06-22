
---

# 🎬 IMDB Sentiment Analysis | Machine Learning & Deep Learning

This project performs **Sentiment Analysis on movie reviews** from the **IMDB Dataset** using both **Traditional Machine Learning** and **Deep Learning** models. The goal is to classify movie reviews as **Positive** or **Negative** with high accuracy.

---

## 📂 Dataset

* **Source:** [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Description:** 50,000 labeled movie reviews, balanced dataset with 25,000 positive and 25,000 negative reviews.

---

## 🚀 Features

✅ **Classical Machine Learning Models (TF-IDF + GridSearchCV):**

* Logistic Regression (Tuned)
* Naive Bayes (Tuned)
* Random Forest (Tuned)

✅ **Deep Learning Models (with Dropout + EarlyStopping to prevent overfitting):**

* RNN
* LSTM
* GRU

✅ **Highlights:**

* Grid Search Hyperparameter Tuning for ML models
* Overfitting Prevention using Dropout + EarlyStopping for DL models
* Model Accuracy Comparison Table and Visualizations
* Automatic Conclusion Generation based on model performance

---

## 🗂️ Folder Structure

```
📦 imdb-sentiment-analysis-nlp-ml-dl
 ┣ 📜 IMDB_Sentiment_Analysis_ML_DL_With_Comparison.ipynb
 ┣ 📜 README.md
 ┣ 📁 /images (optional for plots)
 ┗ 📁 /data (optional if dataset included)
```

---

## 📦 Installation

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

---

## 🖥️ Usage

1. Clone the repository or download the notebook.
2. Run the notebook in:

   * ✅ Google Colab (Recommended for GPU)
   * ✅ Kaggle Notebooks
   * ✅ Jupyter Notebook (Local with TensorFlow GPU/CPU)

---

## 📊 Results Summary

| Model                          | Accuracy |
| ------------------------------ | -------- |
| ✅ Logistic Regression (Tuned)  | 0.8968   |
| LSTM (Dropout + EarlyStopping) | 0.8837   |
| GRU (Dropout + EarlyStopping)  | 0.8802   |
| Naive Bayes (Tuned)            | 0.8580   |
| RNN (Dropout + EarlyStopping)  | 0.8519   |
| Random Forest (Tuned)          | 0.8444   |

---

## 📌 Conclusion

* 📈 **Best Model:** **Logistic Regression (Tuned)** achieved the highest accuracy of **89.68%** on the test dataset.
* Although **deep learning models (LSTM, GRU)** performed **close to Logistic Regression**, they were slightly behind.
* ✅ **Why Logistic Regression performed best:**

  * The IMDB dataset is large and **TF-IDF vectorization creates sparse, high-dimensional features**, making **linear models like Logistic Regression** ideal.
  * Deep Learning models (RNN, LSTM, GRU) **benefit more from word embeddings (e.g., Word2Vec, GloVe)** and larger datasets in general. With TF-IDF, they didn’t show a clear advantage.
* 🏆 **Final Recommendation:** For **TF-IDF feature-based text classification**, **Tuned Logistic Regression is the best choice** in this setup. For further improvement or future work, embeddings + LSTM/GRU can be explored.


---

## ✅ To-Do (Optional Future Improvements)

* [ ] Use **pre-trained word embeddings** (e.g., GloVe, Word2Vec) with deep learning models.
* [ ] Implement **Attention Mechanism** for sequence models.
* [ ] Build a **Streamlit or Gradio web app** for real-time sentiment prediction.
* [ ] Export trained models for deployment (`.h5` or `.pkl`).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any questions, please reach out:

* **Name:** Junaid Akbar
* **Email:** [mja.awan35@gmail.com](mailto:your_email@example.com)
* **GitHub:** [https://github.com/YourGitHubUsername](https://github.com/Junaid-Akbar35)

---

⭐ **If you find this project helpful, please ⭐ star this repository on GitHub!**

---
