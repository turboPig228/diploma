"""
train.py — Обучение двух ML-моделей на датасете account_data.csv

  1. LoanApproval  — одобрение/отказ в кредите
  2. FraudDetection — вероятность мошенничества

Запуск:  python train.py
Вывод:   папка models/ — pkl-файлы моделей, скейлера и энкодеров
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "account_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Категориальные признаки ──────────────────────────────────────────────────
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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def parse_hour(time_str: str) -> int:
    """'10:50' → 10"""
    try:
        return int(str(time_str).split(":")[0])
    except Exception:
        return 12


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки."""
    df = df.copy()
    df["TransactionHour"] = df["TimeofTransaction"].apply(parse_hour)
    safe_income = df["IncomeLevel"].replace(0, 1)
    df["DTI"] = df["LoanAmountRequested"] / safe_income
    return df


def build_loan_label(df: pd.DataFrame) -> pd.Series:
    """
    Правило одобрения кредита (используется только для генерации метки,
    модель потом обучается находить эти паттерны самостоятельно).
    """
    approved = (
        (df["CreditScore"] >= 650)
        & (df["PaymentBehavior"] == "On-time")
        & (df["PastFinancialMalpractices"] == "No")
        & (df["Blacklists"] == "No")
        & (df["EmploymentVerification"] == "Verified")
        & (df["ConsistencyinData"] == "Consistent")
        & (df["DTI"] <= 5.0)
        & (df["IsFraud"] == "No")
    )
    return approved.map({True: 1, False: 0})


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Кодирует CAT_FEATURES с помощью LabelEncoder.
    fit=True  — создаёт и запоминает энкодеры (train)
    fit=False — использует готовые энкодеры (inference)
    """
    df = df.copy()
    if fit:
        encoders = {}
    for col in CAT_FEATURES:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # обрабатываем неизвестные категории
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


# ─── Главная функция обучения ─────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("Загрузка датасета...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Строк: {len(df)}, Столбцов: {len(df.columns)}")

    # ── Признаки ──────────────────────────────────────────────────────────────
    df = build_features(df)
    df["LoanApproved"] = build_loan_label(df)
    df["IsFraud_bin"] = (df["IsFraud"] == "Yes").astype(int)

    print(f"\n  LoanApproved: Yes={df['LoanApproved'].sum()}, No={(df['LoanApproved']==0).sum()}")
    print(f"  IsFraud:      Yes={df['IsFraud_bin'].sum()}, No={(df['IsFraud_bin']==0).sum()}")

    # ── Энкодинг ──────────────────────────────────────────────────────────────
    df_enc, encoders = encode_categoricals(df, fit=True)
    X = df_enc[ALL_FEATURES]

    # ── Скейлинг числовых признаков ───────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUM_FEATURES] = scaler.fit_transform(X[NUM_FEATURES])

    # ═══════════════════════════════════════════════════════════════════════════
    # МОДЕЛЬ 1 — Loan Approval
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Обучение модели Loan Approval (RandomForest)...")

    y_loan = df["LoanApproved"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_loan, test_size=0.2, random_state=42, stratify=y_loan
    )

    loan_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    loan_model.fit(X_tr, y_tr)

    y_pred_loan = loan_model.predict(X_te)
    y_prob_loan = loan_model.predict_proba(X_te)[:, 1]
    print("\nClassification Report (Loan Approval):")
    print(classification_report(y_te, y_pred_loan, target_names=["Rejected", "Approved"]))
    print(f"ROC-AUC: {roc_auc_score(y_te, y_prob_loan):.4f}")

    cv_scores = cross_val_score(loan_model, X_scaled, y_loan, cv=5, scoring="roc_auc")
    print(f"Cross-val ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # МОДЕЛЬ 2 — Fraud Detection
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Обучение модели Fraud Detection (GradientBoosting)...")

    y_fraud = df["IsFraud_bin"]
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_scaled, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )

    fraud_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    fraud_model.fit(X_tr2, y_tr2)

    y_pred_fraud = fraud_model.predict(X_te2)
    y_prob_fraud = fraud_model.predict_proba(X_te2)[:, 1]
    print("\nClassification Report (Fraud Detection):")
    print(classification_report(y_te2, y_pred_fraud, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_te2, y_prob_fraud):.4f}")

    cv_scores2 = cross_val_score(fraud_model, X_scaled, y_fraud, cv=5, scoring="roc_auc")
    print(f"Cross-val ROC-AUC: {cv_scores2.mean():.4f} ± {cv_scores2.std():.4f}")

    # ── Feature importance (топ-10) ───────────────────────────────────────────
    print("\nТоп-10 важных признаков (Fraud Detection):")
    fi = pd.Series(fraud_model.feature_importances_, index=ALL_FEATURES)
    print(fi.nlargest(10).to_string())

    # ── Сохранение ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Сохранение артефактов...")

    joblib.dump(loan_model, os.path.join(MODEL_DIR, "loan_model.pkl"))
    joblib.dump(fraud_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(ALL_FEATURES, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print("  models/loan_model.pkl      — модель одобрения кредита")
    print("  models/fraud_model.pkl     — модель обнаружения мошенничества")
    print("  models/scaler.pkl          — StandardScaler")
    print("  models/encoders.pkl        — LabelEncoders для категорий")
    print("  models/feature_names.pkl   — порядок признаков")
    print("\nОбучение завершено успешно!")


if __name__ == "__main__":
    train()
