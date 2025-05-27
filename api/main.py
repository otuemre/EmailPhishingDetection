from fastapi import FastAPI, Form
from api.pipeline import predict_email
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def root():
    return {'Hello': 'World'}


@app.post('/predict')
def predict(
        sender: str = Form(...),
        receiver: str = Form(' '),
        date: str = Form(' '),
        subject: str = Form(...),
        body: str = Form(...),
        urls: str = Form(' '),
        model_choice: str = Form(...)
):

    prediction = predict_email(sender, receiver, date, subject, body,
                               urls, model_choice=model_choice)

    return {'result': prediction}
