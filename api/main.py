from pipeline import predict_email

from fastapi import FastAPI, Query

app = FastAPI()

@app.get('/')
def root():
    return {'Hello': 'World'}

@app.post('/predict')
def predict(sender: str = '',
            receiver: str = '',
            date: str = '',
            subject:str = '', body: str = '', urls: str = '',
            model_choice: str = Query(..., description="Choose from: 'naive_bayes', 'logistic_regression', 'svm'")
            ) -> dict[str, str]:

    prediction = predict_email(sender=sender, receiver=receiver, date=date, subject=subject,
                        body=body, urls=urls, model_choice=model_choice)

    return {'result': prediction}