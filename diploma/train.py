"""
train.py — Обучение трёх ML-моделей на датасете account_data.csv

  1. LoanApproval (Random Forest)     — одобрение/отказ в кредите
  2. FraudDetection (Gradient Boosting) — вероятность мошенничества
  3. XGBoost (сравнительная модель)   — объединённая задача

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
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "account_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Признаки ─────────────────────────────────────────────────────────────────
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

def parse_hour(time_str):
    try:
        return int(str(time_str).split(":")[0])
    except Exception:
        return 12


def build_features(df):
    df = df.copy()
    df["TransactionHour"] = df["TimeofTransaction"].apply(parse_hour)
    df["DTI"] = df["LoanAmountRequested"] / df["IncomeLevel"].replace(0, 1)
    return df


def build_loan_label(df):
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


def encode_categoricals(df, encoders=None, fit=True):
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
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─── Главная функция ──────────────────────────────────────────────────────────

def train():
    print_separator("Загрузка и подготовка данных")
    df = pd.read_csv(DATA_PATH)
    print(f"  Строк: {len(df)}, Столбцов: {len(df.columns)}")

    df = build_features(df)
    df["LoanApproved"] = build_loan_label(df)
    df["IsFraud_bin"]  = (df["IsFraud"] == "Yes").astype(int)

    print(f"  LoanApproved: Yes={df['LoanApproved'].sum()}, No={(df['LoanApproved']==0).sum()}")
    print(f"  IsFraud:      Yes={df['IsFraud_bin'].sum()}, No={(df['IsFraud_bin']==0).sum()}")

    df_enc, encoders = encode_categoricals(df, fit=True)
    X = df_enc[ALL_FEATURES]

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUM_FEATURES] = scaler.fit_transform(X[NUM_FEATURES])

    # Таблица сравнения
    comparison = []

    # ═══════════════════════════════════════════════════════════════════════
    # МОДЕЛЬ 1 — Loan Approval (Random Forest)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("Модель 1 — Loan Approval (Random Forest)")

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

    y_pred = loan_model.predict(X_te)
    y_prob = loan_model.predict_proba(X_te)[:, 1]
    auc_rf  = roc_auc_score(y_te, y_prob)
    acc_rf  = accuracy_score(y_te, y_pred)
    cv_rf   = cross_val_score(loan_model, X_scaled, y_loan, cv=5, scoring="roc_auc").mean()

    print(classification_report(y_te, y_pred, target_names=["Rejected", "Approved"]))
    print(f"  ROC-AUC:         {auc_rf:.4f}")
    print(f"  Accuracy:        {acc_rf:.4f}")
    print(f"  Cross-val AUC:   {cv_rf:.4f}")
    comparison.append({"Модель": "Random Forest (Loan)", "ROC-AUC": auc_rf, "Accuracy": acc_rf, "CV AUC": cv_rf})

    # ═══════════════════════════════════════════════════════════════════════
    # МОДЕЛЬ 2 — Fraud Detection (Gradient Boosting)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("Модель 2 — Fraud Detection (Gradient Boosting)")

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

    y_pred2 = fraud_model.predict(X_te2)
    y_prob2 = fraud_model.predict_proba(X_te2)[:, 1]
    auc_gb  = roc_auc_score(y_te2, y_prob2)
    acc_gb  = accuracy_score(y_te2, y_pred2)
    cv_gb   = cross_val_score(fraud_model, X_scaled, y_fraud, cv=5, scoring="roc_auc").mean()

    print(classification_report(y_te2, y_pred2, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC:         {auc_gb:.4f}")
    print(f"  Accuracy:        {acc_gb:.4f}")
    print(f"  Cross-val AUC:   {cv_gb:.4f}")
    comparison.append({"Модель": "Gradient Boosting (Fraud)", "ROC-AUC": auc_gb, "Accuracy": acc_gb, "CV AUC": cv_gb})

    # ═══════════════════════════════════════════════════════════════════════
    # МОДЕЛЬ 3 — XGBoost (сравнительная, обе задачи)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("Модель 3 — XGBoost (сравнительная)")

    # XGBoost на задаче одобрения кредита
    print("\n  [XGBoost → Loan Approval]")
    xgb_loan = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    xgb_loan.fit(X_tr, y_tr)

    y_pred_xl = xgb_loan.predict(X_te)
    y_prob_xl = xgb_loan.predict_proba(X_te)[:, 1]
    auc_xl    = roc_auc_score(y_te, y_prob_xl)
    acc_xl    = accuracy_score(y_te, y_pred_xl)
    cv_xl     = cross_val_score(xgb_loan, X_scaled, y_loan, cv=5, scoring="roc_auc").mean()

    print(classification_report(y_te, y_pred_xl, target_names=["Rejected", "Approved"]))
    print(f"  ROC-AUC:         {auc_xl:.4f}")
    print(f"  Accuracy:        {acc_xl:.4f}")
    print(f"  Cross-val AUC:   {cv_xl:.4f}")
    comparison.append({"Модель": "XGBoost (Loan)", "ROC-AUC": auc_xl, "Accuracy": acc_xl, "CV AUC": cv_xl})

    # XGBoost на задаче фрода
    print("\n  [XGBoost → Fraud Detection]")
    xgb_fraud = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=int((y_fraud == 0).sum() / max((y_fraud == 1).sum(), 1)),
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    xgb_fraud.fit(X_tr2, y_tr2)

    y_pred_xf = xgb_fraud.predict(X_te2)
    y_prob_xf = xgb_fraud.predict_proba(X_te2)[:, 1]
    auc_xf    = roc_auc_score(y_te2, y_prob_xf)
    acc_xf    = accuracy_score(y_te2, y_pred_xf)
    cv_xf     = cross_val_score(xgb_fraud, X_scaled, y_fraud, cv=5, scoring="roc_auc").mean()

    print(classification_report(y_te2, y_pred_xf, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC:         {auc_xf:.4f}")
    print(f"  Accuracy:        {acc_xf:.4f}")
    print(f"  Cross-val AUC:   {cv_xf:.4f}")
    comparison.append({"Модель": "XGBoost (Fraud)", "ROC-AUC": auc_xf, "Accuracy": acc_xf, "CV AUC": cv_xf})

    # ═══════════════════════════════════════════════════════════════════════
    # СРАВНИТЕЛЬНАЯ ТАБЛИЦА
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("Сравнительная таблица моделей")
    df_cmp = pd.DataFrame(comparison)
    df_cmp["ROC-AUC"] = df_cmp["ROC-AUC"].map("{:.4f}".format)
    df_cmp["Accuracy"] = df_cmp["Accuracy"].map("{:.4f}".format)
    df_cmp["CV AUC"]  = df_cmp["CV AUC"].map("{:.4f}".format)
    print(df_cmp.to_string(index=False))

    # ─── Топ-10 важных признаков XGBoost ──────────────────────────────────
    print("\n  Топ-10 признаков (XGBoost Loan):")
    fi = pd.Series(xgb_loan.feature_importances_, index=ALL_FEATURES)
    print(fi.nlargest(10).to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # СОХРАНЕНИЕ
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("Сохранение артефактов")

    joblib.dump(loan_model,  os.path.join(MODEL_DIR, "loan_model.pkl"))
    joblib.dump(fraud_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(xgb_loan,    os.path.join(MODEL_DIR, "xgb_loan_model.pkl"))
    joblib.dump(xgb_fraud,   os.path.join(MODEL_DIR, "xgb_fraud_model.pkl"))
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders,    os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(ALL_FEATURES,os.path.join(MODEL_DIR, "feature_names.pkl"))

    print("  loan_model.pkl        — Random Forest (Loan Approval)")
    print("  fraud_model.pkl       — Gradient Boosting (Fraud Detection)")
    print("  xgb_loan_model.pkl    — XGBoost (Loan Approval)")
    print("  xgb_fraud_model.pkl   — XGBoost (Fraud Detection)")
    print("  scaler.pkl            — StandardScaler")
    print("  encoders.pkl          — LabelEncoders")
    print("\nОбучение завершено!")


if __name__ == "__main__":
    train()