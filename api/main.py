from api.pipeline import predict_email

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

from pydantic import BaseModel

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
    return FileResponse("frontend/static/NotFound.html", status_code=404)


@app.get("/robots.txt", include_in_schema=False)
def robots():
    return FileResponse("frontend/static/robots.txt", media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap():
    return FileResponse("frontend/static/sitemap.xml", media_type="application/xml")


@app.get("/google4c935f6316549eef.html")
def verify_google():
    return FileResponse("frontend/static/google4c935f6316549eef.html")


@app.get('/')
def root():
    return FileResponse('frontend/static/index.html')


class EmailInput(BaseModel):
    sender: str
    receiver: str = ""
    date: str = ""
    subject: str
    body: str
    urls: str = ""
    model_choice: str


@app.post('/predict')
def predict(input: EmailInput):
    # Get the email verdict
    email_result = predict_email(
        sender=input.sender,
        receiver=input.receiver,
        date=input.date,
        subject=input.subject,
        body=input.body,
        urls=input.urls,
        model_choice=input.model_choice
    )

    return {'result': email_result}
