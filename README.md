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
- [Future Improvements](#-future-improvements)
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

### 🚀 Future Improvements

- **📡 Public API & Documentation**  
  Provide a proper REST API endpoint with OpenAPI/Swagger documentation so developers can integrate the phishing detection system into their own applications.

- **🎨 Improve the UI**  
  Rebuild the frontend using a modern framework like React (possibly with Tailwind or Material UI) to create a more interactive and responsive user experience.

- **🔗 URL-Based Model**  
  Train and integrate a secondary model focused specifically on analyzing URLs for phishing characteristics such as domain structure, length, obfuscation, and suspicious keywords.

- **📈 Expand the Dataset**  
  Enhance the model's performance by collecting a larger and more diverse dataset of phishing and legitimate emails, improving generalization and reducing bias.

- **🧠 Improve Model Explainability**  
  Integrate explainable AI tools like SHAP or LIME to provide transparency into why the model classified an email as phishing or legitimate.

- **📬 Real-Time Email API Integration (Optional)**  
  Integrate with email providers like Gmail or Microsoft Outlook via API to allow live scanning of user inboxes (with permission) and flag suspicious messages in real time.

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
