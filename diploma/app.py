"""
app.py — FastAPI сервер для инференса ML-моделей

Эндпоинты:
  GET  /health        — проверка состояния
  POST /predict       — полный анализ: одобрение кредита + риск мошенничества
  GET  /model-info    — информация о загруженных моделях

Запуск:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Константы из train.py ───────────────────────────────────────────────────
CAT_FEATURES = [
    "Occupation", "MaritalStatus", "ResidentialStatus", "PurposeoftheLoan",
    "Collateral", "ApplicationBehavior", "LocationofApplication",
    "ChangeinBehavior", "AccountActivity", "PaymentBehavior",
    "Blacklists", "EmploymentVerification", "PastFinancialMalpractices",
    "DeviceInformation", "SocialMediaFootprint", "ConsistencyinData", "Referral",
]
NUM_FEATURES = [
    "Age", "Dependents", "AddressDuration", "CreditScore", "IncomeLevel",
    "LoanAmountRequested", "LoanTerm", "InterestRate", "PreviousLoans",
    "ExistingLiabilities", "TransactionHour", "DTI",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# ─── Загрузка артефактов ──────────────────────────────────────────────────────
def load_artifacts():
    try:
        return {
            "loan_model":   joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl")),
            "fraud_model":  joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl")),
            "scaler":       joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            "encoders":     joblib.load(os.path.join(MODEL_DIR, "encoders.pkl")),
        }
    except FileNotFoundError as e:
        raise RuntimeError(f"Модели не найдены. Запустите train.py сначала. {e}")

artifacts = load_artifacts()

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Approval & Fraud Detection API",
    description="DSS — система поддержки принятия решений по кредитованию",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Схема входящего запроса ──────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    # Демографические данные
    age: int                  = Field(..., ge=18, le=100, description="Возраст заявителя")
    occupation: str           = Field(..., description="Профессия")
    marital_status: str       = Field(..., description="Семейное положение: Single / Married / Divorced")
    dependents: int           = Field(..., ge=0, le=20, description="Кол-во иждивенцев")
    residential_status: str   = Field(..., description="Статус жилья: Own / Rent / Live with Parents")
    address_duration: int     = Field(..., ge=0, description="Срок проживания по адресу (мес.)")

    # Финансовые данные
    credit_score: int         = Field(..., ge=300, le=850, description="Кредитный рейтинг")
    income_level: float       = Field(..., gt=0, description="Уровень дохода (в месяц)")
    loan_amount_requested: float = Field(..., gt=0, description="Запрашиваемая сумма кредита")
    loan_term: int            = Field(..., ge=1, le=360, description="Срок кредита (мес.)")
    purpose_of_loan: str      = Field(..., description="Цель: home / auto / personal / education / travel / medical")
    collateral: str           = Field(..., description="Залог: Yes / No")
    interest_rate: float      = Field(..., ge=0, le=100, description="Процентная ставка (%)")
    previous_loans: int       = Field(..., ge=0, description="Количество прошлых кредитов")
    existing_liabilities: float = Field(..., ge=0, description="Текущие обязательства")

    # Поведенческие признаки
    application_behavior: str     = Field(..., description="Поведение при подаче: Normal / Rapid")
    location_of_application: str  = Field(..., description="Местоположение: Local / Unusual")
    change_in_behavior: str       = Field(..., description="Изменение поведения: Yes / No")
    time_of_transaction: str      = Field(..., description="Время транзакции (HH:MM), напр. 14:30")
    account_activity: str         = Field(..., description="Активность счёта: Normal / Unusual")
    payment_behavior: str         = Field(..., description="История платежей: On-time / Late / Defaulted")

    # Верификация и риски
    blacklists: str                    = Field(..., description="Наличие в чёрных списках: Yes / No")
    employment_verification: str       = Field(..., description="Верификация занятости: Verified / Not Verified")
    past_financial_malpractices: str   = Field(..., description="Финансовые нарушения в прошлом: Yes / No")

    # Технические данные
    device_information: str     = Field(..., description="Устройство: Mobile / Laptop / Desktop / Tablet")
    social_media_footprint: str = Field(..., description="Присутствие в соцсетях: Yes / No")
    consistency_in_data: str    = Field(..., description="Согласованность данных: Consistent / Inconsistent")
    referral: str               = Field(..., description="Источник: Referral / Online")

    @field_validator("time_of_transaction")
    @classmethod
    def validate_time(cls, v):
        try:
            parts = v.split(":")
            assert 0 <= int(parts[0]) <= 23
            assert 0 <= int(parts[1]) <= 59
        except Exception:
            raise ValueError("Неверный формат времени. Используйте HH:MM, например 14:30")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35, "occupation": "Engineer", "marital_status": "Married",
            "dependents": 1, "residential_status": "Own", "address_duration": 36,
            "credit_score": 780, "income_level": 75000, "loan_amount_requested": 200000,
            "loan_term": 24, "purpose_of_loan": "home", "collateral": "Yes",
            "interest_rate": 5.5, "previous_loans": 1, "existing_liabilities": 5000,
            "application_behavior": "Normal", "location_of_application": "Local",
            "change_in_behavior": "No", "time_of_transaction": "10:30",
            "account_activity": "Normal", "payment_behavior": "On-time",
            "blacklists": "No", "employment_verification": "Verified",
            "past_financial_malpractices": "No", "device_information": "Laptop",
            "social_media_footprint": "Yes", "consistency_in_data": "Consistent",
            "referral": "Online",
        }
    }}


# ─── Схема ответа ─────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    request_id:            str
    timestamp:             str

    # Кредитное решение
    loan_decision:         Literal["Approved", "Rejected"]
    loan_approval_prob:    float  # вероятность одобрения [0..1]
    rejection_reasons:     list[str]

    # Антифрод
    fraud_risk:            Literal["Low", "Medium", "High"]
    fraud_probability:     float  # вероятность мошенничества [0..1]
    fraud_flags:           list[str]

    # Агрегированное решение
    final_decision:        Literal["Approved", "Rejected", "Review Required"]
    confidence_score:      float
    dti_ratio:             float


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def request_to_dataframe(req: PredictionRequest) -> pd.DataFrame:
    """Превращает Pydantic-модель в DataFrame с нужными именами столбцов."""
    hour = int(req.time_of_transaction.split(":")[0])
    safe_income = req.income_level if req.income_level > 0 else 1
    dti = req.loan_amount_requested / safe_income

    row = {
        # числовые
        "Age":                  req.age,
        "Dependents":           req.dependents,
        "AddressDuration":      req.address_duration,
        "CreditScore":          req.credit_score,
        "IncomeLevel":          req.income_level,
        "LoanAmountRequested":  req.loan_amount_requested,
        "LoanTerm":             req.loan_term,
        "InterestRate":         req.interest_rate,
        "PreviousLoans":        req.previous_loans,
        "ExistingLiabilities":  req.existing_liabilities,
        "TransactionHour":      hour,
        "DTI":                  dti,
        # категориальные
        "Occupation":                   req.occupation,
        "MaritalStatus":                req.marital_status,
        "ResidentialStatus":            req.residential_status,
        "PurposeoftheLoan":             req.purpose_of_loan,
        "Collateral":                   req.collateral,
        "ApplicationBehavior":          req.application_behavior,
        "LocationofApplication":        req.location_of_application,
        "ChangeinBehavior":             req.change_in_behavior,
        "AccountActivity":              req.account_activity,
        "PaymentBehavior":              req.payment_behavior,
        "Blacklists":                   req.blacklists,
        "EmploymentVerification":       req.employment_verification,
        "PastFinancialMalpractices":    req.past_financial_malpractices,
        "DeviceInformation":            req.device_information,
        "SocialMediaFootprint":         req.social_media_footprint,
        "ConsistencyinData":            req.consistency_in_data,
        "Referral":                     req.referral,
    }
    return pd.DataFrame([row])


def encode_and_scale(df: pd.DataFrame) -> np.ndarray:
    """Энкодинг + скейлинг — воспроизводит логику train.py."""
    encoders = artifacts["encoders"]
    scaler   = artifacts["scaler"]

    for col in CAT_FEATURES:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    X = df[ALL_FEATURES].copy()
    X[NUM_FEATURES] = scaler.transform(X[NUM_FEATURES])
    return X.values


def get_rejection_reasons(req: PredictionRequest, loan_prob: float) -> list[str]:
    reasons = []
    if req.credit_score < 650:
        reasons.append(f"Кредитный рейтинг ниже порога ({req.credit_score} < 650)")
    if req.payment_behavior != "On-time":
        reasons.append(f"Неблагоприятная история платежей: {req.payment_behavior}")
    if req.past_financial_malpractices == "Yes":
        reasons.append("Зафиксированы прошлые финансовые нарушения")
    if req.blacklists == "Yes":
        reasons.append("Заявитель в чёрном списке")
    if req.employment_verification == "Not Verified":
        reasons.append("Занятость не подтверждена")
    if req.consistency_in_data == "Inconsistent":
        reasons.append("Несогласованность предоставленных данных")
    dti = req.loan_amount_requested / max(req.income_level, 1)
    if dti > 5.0:
        reasons.append(f"Высокий показатель DTI: {dti:.1f} (норма ≤ 5.0)")
    return reasons


def get_fraud_flags(req: PredictionRequest) -> list[str]:
    flags = []
    if req.application_behavior == "Rapid":
        flags.append("Быстрая подача заявки (подозрительная скорость)")
    if req.location_of_application == "Unusual":
        flags.append("Необычное местоположение при подаче")
    if req.change_in_behavior == "Yes":
        flags.append("Зафиксировано изменение поведения")
    if req.account_activity == "Unusual":
        flags.append("Нестандартная активность по счёту")
    if req.consistency_in_data == "Inconsistent":
        flags.append("Несоответствия в предоставленных данных")
    hour = int(req.time_of_transaction.split(":")[0])
    if 0 <= hour <= 5:
        flags.append(f"Транзакция в нетипичное время ({req.time_of_transaction})")
    return flags


def fraud_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Medium"
    return "High"


# ─── Эндпоинты ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": ["loan_model", "fraud_model"],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model-info")
def model_info():
    loan_m  = artifacts["loan_model"]
    fraud_m = artifacts["fraud_model"]
    return {
        "loan_model": {
            "type": type(loan_m).__name__,
            "n_estimators": getattr(loan_m, "n_estimators", None),
            "features": len(ALL_FEATURES),
        },
        "fraud_model": {
            "type": type(fraud_m).__name__,
            "n_estimators": getattr(fraud_m, "n_estimators", None),
            "features": len(ALL_FEATURES),
        },
        "feature_list": ALL_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        df = request_to_dataframe(req)
        X  = encode_and_scale(df)   # returns np.ndarray; wrap back for named features
        X_df = pd.DataFrame(X, columns=ALL_FEATURES)

        # Предсказания
        loan_prob  = float(artifacts["loan_model"].predict_proba(X_df)[0][1])
        fraud_prob = float(artifacts["fraud_model"].predict_proba(X_df)[0][1])

        loan_decision = "Approved" if loan_prob >= 0.5 else "Rejected"
        f_risk = fraud_level(fraud_prob)

        # Финальное решение с учётом фрода
        if loan_decision == "Approved" and f_risk == "High":
            final = "Review Required"
        elif loan_decision == "Approved" and f_risk == "Medium":
            final = "Review Required"
        else:
            final = loan_decision

        dti = req.loan_amount_requested / max(req.income_level, 1)
        confidence = round(abs(loan_prob - 0.5) * 2, 3)  # [0..1]

        return PredictionResponse(
            request_id=f"REQ-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}",
            timestamp=datetime.utcnow().isoformat(),
            loan_decision=loan_decision,
            loan_approval_prob=round(loan_prob, 4),
            rejection_reasons=get_rejection_reasons(req, loan_prob) if loan_decision == "Rejected" else [],
            fraud_risk=f_risk,
            fraud_probability=round(fraud_prob, 4),
            fraud_flags=get_fraud_flags(req),
            final_decision=final,
            confidence_score=confidence,
            dti_ratio=round(dti, 3),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
