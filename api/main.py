from fastapi import FastAPI, Form
from api.pipeline import predict_email
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.requests import Request
from starlette.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://phishingdetection.net"],  # Make it ['*'] for local testing
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory='frontend/static'), name='static')


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    return FileResponse("frontend/static/404.html", status_code=404)


@app.get('/')
def root():
    return FileResponse('frontend/static/index.html')


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
