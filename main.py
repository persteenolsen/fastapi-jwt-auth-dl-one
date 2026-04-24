import os
import numpy as np
import onnxruntime as ort

from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel, field_validator
from jose import jwt, JWTError
from dotenv import load_dotenv

# -----------------------
# ENV
# -----------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "password")


# -----------------------------
# INIT APP
# -----------------------------
# app = FastAPI()
app = FastAPI(
    title="FastAPI + JWT + Deep Learning + XOR Neural Network",
    description="24-04-2026 - FastAPI + JWT + Deep Learning + XOR Neural Network trained by PyTorch and exported to ONNX",
    version="1.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -----------------------
# MODEL LOAD
# -----------------------
MODEL_PATH = "model.onnx"
session = ort.InferenceSession(MODEL_PATH)

# -----------------------
# GLOBAL ERROR HANDLERS
# -----------------------

# Invalid JSON / validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "message": "Send valid JSON like: {\"x1\": 0, \"x2\": 1}"
        }
    )

# Generic HTTP errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail
        }
    )

# -----------------------
# JWT AUTH
# -----------------------
def create_token(username: str):
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -----------------------
# INPUT MODEL (STRICT XOR)
# -----------------------
class InputData(BaseModel):
    x1: float
    x2: float

    @field_validator("x1", "x2")
    @classmethod
    def validate_binary(cls, v):
        # Reject non-numeric types
        if not isinstance(v, (int, float)):
            raise ValueError("Must be a number (0 or 1)")

        # XOR restriction
        if v not in (0, 1, 0.0, 1.0):
            raise ValueError("XOR model only accepts 0 or 1")

        return float(v)

# -----------------------
# ROOT ENDPOINT
# -----------------------
@app.get("/")
def root():
    return {"message": "Deep Learning API running"}

# -----------------------
# AUTH ENDPOINT
# -----------------------
@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != USERNAME or form.password != PASSWORD:
        raise HTTPException(status_code=401, detail="Bad credentials")

    return {
        "access_token": create_token(form.username),
        "token_type": "bearer"
    }

# -----------------------
# PREDICTION ENDPOINT
# -----------------------
@app.post("/predict")
def predict(data: InputData, user: str = Depends(get_user)):

    input_array = np.array([[data.x1, data.x2]], dtype=np.float32)

    result = session.run(
        None,
        {"input": input_array}
    )

    prediction = float(result[0][0][0])

    return {
        "user": user,
        "prediction": prediction
    }