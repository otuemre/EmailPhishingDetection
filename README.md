# 📧 Email Phishing Detection

[![License: MIT](https://img.shields.io/github/license/otuemre/EmailPhishingDetection?style=flat-square)](./LICENSE.md)
[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-5e60ce?logo=render&style=flat-square)](https://phishingdetection.net)
[![Hugging Face Models](https://img.shields.io/badge/HuggingFace-SVM%20%7C%20TFIDF-orange?logo=huggingface&style=flat-square)](https://huggingface.co/emreotu)

Detect phishing emails in real-time using machine learning — trained on six merged datasets and deployed via a full-stack FastAPI app.

🔗 **Live Demo**: [https://phishingdetection.net](https://phishingdetection.net)

## 📚 Table of Contents

- [What It Does](#-what-it-does)
- [Tech Stack](#️-tech-stack)
- [ML Models](#-ml-models)
- [Sample Input](#-sample-input)
- [Project Structure](#-project-structure)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 🧠 What It Does

This project allows users to paste real email content (sender, subject, body, etc.) and choose between three machine learning models to detect whether it's **Phishing** or **Legitimate**.

It combines:
- 📊 Natural Language Processing (TF-IDF + NLTK)
- 🤖 ML models (Naive Bayes, Logistic Regression, SVM)
- 🌍 Live API deployment + responsive UI

---

## 🛠️ Tech Stack

| Layer          | Tech                       |
|----------------|----------------------------|
| Frontend       | HTML, CSS, JavaScript      |
| Backend        | FastAPI, Uvicorn           |
| ML/NLP         | Scikit-learn, NLTK, joblib |
| Deployment     | Render, Namecheap          |
| Hosting Models | Hugging Face 🤗            |

---

## 🤖 ML Models

Choose between:
- ✅ Support Vector Machine (Best Accuracy)
- ✅ Logistic Regression
- ✅ Multinomial Naive Bayes

### 📦 Hugging Face Models

- 🔗 [SVM Model](https://huggingface.co/otuemre/email-phishing-svm)
- 🔗 [TF-IDF Vectorizer](https://huggingface.co/otuemre/email-phishing-vectorizer)

---

## 🧪 Sample Input

```
Sender: freeiphone@gmail.com
Subject: Don't miss this chance!
Body: Click the link to claim your free iPhone 14 Pro Max.
Date: May 5, 2025
Model: SVM
```

✔️ Output: **Phishing**

---

## 🗂️ Project Structure

```
EmailPhishingDetection/
├── api/
│   ├── main.py
│   └── pipeline.py
├── data/
│   └── phishing_email.csv
├── frontend/
│   └── static/
│       └── index.html
├── models/
│   ├── logistic_regression_model.joblib
│   ├── naive_bayes_model.joblib
│   ├── svm_model.joblib
│   └── tfidf_vectorizer.joblib
├── notebooks/
│   └── 01_training.ipynb
├── images/
├── .gitignore
├── LICENSE.md
├── README.md
├── render.yml
└── requirements.txt
```

---

## 🙏 Acknowledgements

This project uses the **Phishing Email Dataset** by [Naser Abdullah Alam on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset).

Please cite the following article if using this dataset:

> **Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19).**  
> *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection*.  
> ArXiv: [https://arxiv.org/abs/2405.11619](https://arxiv.org/abs/2405.11619)

---

## 👨‍💻 Author

**Emre OTU**  
🔗 [GitHub](https://github.com/otuemre) | [LinkedIn](https://linkedin.com/in/emreotu)
