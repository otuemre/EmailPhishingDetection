import re

import joblib
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Vectorizer and All the Models
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
models = {
    'naive_bayes': joblib.load(MODEL_DIR / "naive_bayes_model.joblib"),
    'logistic_regression': joblib.load(MODEL_DIR / "logistic_regression_model.joblib"),
    'svm': joblib.load(MODEL_DIR / "svm_model.joblib"),
}

# Merging the Fields
def merge_fields(sender: str = '', receiver: str = '', date: str = '',
                 subject:str = '', body: str = '', urls: str = '') -> str:

    # To safely merge
    fields = [sender, receiver, date, subject, body, urls]
    # Merge and Return the Combined Text
    return " ".join(str(field).strip() for field in fields if field)

# Text Preprocessing
def preprocess_text(text: str) -> str:
    # To Lower Case
    text = text.lower()

    # Remove Digits
    text = re.sub(r'\d+', '', text)

    # Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove Whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Tokenize and Remove the Stop Words
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in filtered]

    # Return the Preprocessed Text
    return ' '.join(lemmatized)

def predict_email(sender: str = '', receiver: str = '', date: str = '',
                  subject:str = '', body: str = '', urls: str = '',
                  model_choice: str = '') -> str:

    # Step 1: Merge the Text
    combined_text = merge_fields(sender, receiver, date, subject, body, urls)

    # Step 2: Preprocess the Text
    clean_text = preprocess_text(combined_text)

    # Step 3: Vectorize the Text
    vector = vectorizer.transform([clean_text])

    # Step 4: Select the Model and Predict the Result
    if model_choice not in models:
        raise ValueError("Invalid Model Choice! Please Select From: 'naive_bayes', 'logistic_regression', 'svm'")

    model = models[model_choice]
    prediction = model.predict(vector)[0]

    return "Phishing" if prediction == 1 else "Legitimate"

if  __name__ == "__main__":
    sender = 'googledev-noreply@google.com'
    subject = 'X, one week until I/O… have you registered?'
    body = '''
        Hi X,
        Google I/O is a week away and you don’t want to miss it! Join us for two days of product announcements and sessions covering the latest tech. Livestream keynotes start May 20 at 10 am PT.
        Don’t wait! Register today so you’re ready when it all goes live.
    '''
    date = 'Tue 13 May, 17:26'
    model = 'svm'

    res = predict_email(sender=sender, date=date, subject=subject, body=body, model_choice=model)
    print("The Email is", res)
