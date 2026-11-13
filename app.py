"""Churn Prediction App (Tugas Besar Part 3).

Menjalankan model Random Forest hasil tuning (Part 2) lewat API FastAPI,
sekaligus menyajikan halaman web (index.html) sebagai User Interface.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}. "
        "Unduh churn_model.pth dari Google Colab lalu letakkan satu folder dengan app.py."
    )

# File .pth ini berisi pipeline scikit-learn (preprocessing + Random Forest) yang disimpan dengan joblib
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    """Halaman User Interface."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest (tuned)"}


@app.post("/predict")
def predict(customer: Customer):
    data = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    prob = float(model.predict_proba(pd.DataFrame([data]))[0, 1])
    return {
        "churn_probability": round(prob, 4),
        "prediction": "Churn" if prob >= 0.5 else "Tidak Churn",
    }
