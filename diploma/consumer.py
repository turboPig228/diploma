"""
consumer.py — ML-воркер: Kafka Consumer + Producer

Поток данных:
  DSS (Node.js) → topic: loan-ml-input  →  [этот файл]  → topic: loan-ml-output
                                                          → topic: loan-ml-logs

Запуск:
  python consumer.py

Переменные окружения (.env):
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092
  KAFKA_INPUT_TOPIC=loan-ml-input
  KAFKA_OUTPUT_TOPIC=loan-ml-output
  KAFKA_LOGS_TOPIC=loan-ml-logs
  KAFKA_GROUP_ID=ml-consumer-group
"""

import os
import json
import time
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from dotenv import load_dotenv

load_dotenv()

# ─── Логирование ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml-worker")

# ─── Конфигурация из .env ─────────────────────────────────────────────────────
KAFKA_SERVERS    = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC      = os.getenv("KAFKA_INPUT_TOPIC",  "loan-ml-input")
OUTPUT_TOPIC     = os.getenv("KAFKA_OUTPUT_TOPIC", "loan-ml-output")
LOGS_TOPIC       = os.getenv("KAFKA_LOGS_TOPIC",   "loan-ml-logs")
GROUP_ID         = os.getenv("KAFKA_GROUP_ID",     "ml-consumer-group")
RETRY_INTERVAL   = int(os.getenv("KAFKA_RETRY_INTERVAL", "5"))   # сек между попытками
MAX_RETRIES      = int(os.getenv("KAFKA_MAX_RETRIES", "10"))

# ─── Пути к моделям ──────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Те же константы, что в train.py и app.py ────────────────────────────────
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

def load_artifacts() -> dict:
    log.info("Загрузка ML-артефактов из %s", MODEL_DIR)
    artifacts = {
        "loan_model":  joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl")),
        "fraud_model": joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl")),
        "scaler":      joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "encoders":    joblib.load(os.path.join(MODEL_DIR, "encoders.pkl")),
    }
    log.info("Модели загружены: %s", list(artifacts.keys()))
    return artifacts


# ─── Preprocessing (повторяет логику app.py) ─────────────────────────────────

def parse_hour(time_str: str) -> int:
    try:
        return int(str(time_str).split(":")[0])
    except Exception:
        return 12


def preprocess(data: dict, artifacts: dict) -> pd.DataFrame:
    """
    Принимает сырой dict из Kafka-сообщения,
    возвращает готовый DataFrame для predict_proba.
    """
    safe_income = float(data.get("income_level", 1) or 1)
    dti = float(data.get("loan_amount_requested", 0)) / safe_income

    row = {
        # числовые
        "Age":                 int(data.get("age", 0)),
        "Dependents":          int(data.get("dependents", 0)),
        "AddressDuration":     int(data.get("address_duration", 0)),
        "CreditScore":         float(data.get("credit_score", 600)),
        "IncomeLevel":         float(data.get("income_level", 0)),
        "LoanAmountRequested": float(data.get("loan_amount_requested", 0)),
        "LoanTerm":            int(data.get("loan_term", 12)),
        "InterestRate":        float(data.get("interest_rate", 5)),
        "PreviousLoans":       int(data.get("previous_loans", 0)),
        "ExistingLiabilities": float(data.get("existing_liabilities", 0)),
        "TransactionHour":     parse_hour(data.get("time_of_transaction", "12:00")),
        "DTI":                 dti,
        # категориальные
        "Occupation":               str(data.get("occupation", "")),
        "MaritalStatus":            str(data.get("marital_status", "")),
        "ResidentialStatus":        str(data.get("residential_status", "")),
        "PurposeoftheLoan":         str(data.get("purpose_of_loan", "")),
        "Collateral":               str(data.get("collateral", "No")),
        "ApplicationBehavior":      str(data.get("application_behavior", "Normal")),
        "LocationofApplication":    str(data.get("location_of_application", "Local")),
        "ChangeinBehavior":         str(data.get("change_in_behavior", "No")),
        "AccountActivity":          str(data.get("account_activity", "Normal")),
        "PaymentBehavior":          str(data.get("payment_behavior", "On-time")),
        "Blacklists":               str(data.get("blacklists", "No")),
        "EmploymentVerification":   str(data.get("employment_verification", "Verified")),
        "PastFinancialMalpractices":str(data.get("past_financial_malpractices", "No")),
        "DeviceInformation":        str(data.get("device_information", "Laptop")),
        "SocialMediaFootprint":     str(data.get("social_media_footprint", "No")),
        "ConsistencyinData":        str(data.get("consistency_in_data", "Consistent")),
        "Referral":                 str(data.get("referral", "Online")),
    }

    df = pd.DataFrame([row])
    encoders = artifacts["encoders"]

    for col in CAT_FEATURES:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])

    df[NUM_FEATURES] = artifacts["scaler"].transform(df[NUM_FEATURES])
    return df[ALL_FEATURES]


# ─── Бизнес-логика решения ───────────────────────────────────────────────────

def fraud_level(prob: float) -> str:
    if prob < 0.35:  return "Low"
    if prob < 0.65:  return "Medium"
    return "High"


def get_rejection_reasons(data: dict, loan_prob: float) -> list:
    reasons = []
    cs = float(data.get("credit_score", 0))
    if cs < 650:
        reasons.append(f"Кредитный рейтинг ниже порога ({int(cs)} < 650)")
    if data.get("payment_behavior") != "On-time":
        reasons.append(f"История платежей: {data.get('payment_behavior')}")
    if data.get("past_financial_malpractices") == "Yes":
        reasons.append("Прошлые финансовые нарушения")
    if data.get("blacklists") == "Yes":
        reasons.append("Заявитель в чёрном списке")
    if data.get("employment_verification") == "Not Verified":
        reasons.append("Занятость не подтверждена")
    if data.get("consistency_in_data") == "Inconsistent":
        reasons.append("Несогласованность данных")
    income = float(data.get("income_level", 1) or 1)
    dti = float(data.get("loan_amount_requested", 0)) / income
    if dti > 5.0:
        reasons.append(f"Высокий DTI: {dti:.1f} (норма ≤ 5.0)")
    return reasons


def get_fraud_flags(data: dict) -> list:
    flags = []
    if data.get("application_behavior") == "Rapid":
        flags.append("Быстрая подача заявки")
    if data.get("location_of_application") == "Unusual":
        flags.append("Необычное местоположение")
    if data.get("change_in_behavior") == "Yes":
        flags.append("Изменение поведения")
    if data.get("account_activity") == "Unusual":
        flags.append("Нестандартная активность счёта")
    if data.get("consistency_in_data") == "Inconsistent":
        flags.append("Несоответствия в данных")
    hour = parse_hour(data.get("time_of_transaction", "12:00"))
    if 0 <= hour <= 5:
        flags.append(f"Транзакция ночью ({data.get('time_of_transaction')})")
    return flags


def build_result(data: dict, artifacts: dict) -> dict:
    """Запускает оба классификатора и формирует итоговый ответ."""
    X_df = preprocess(data, artifacts)

    loan_prob  = float(artifacts["loan_model"].predict_proba(X_df)[0][1])
    fraud_prob = float(artifacts["fraud_model"].predict_proba(X_df)[0][1])

    loan_decision = "Approved" if loan_prob >= 0.5 else "Rejected"
    f_risk = fraud_level(fraud_prob)

    if loan_decision == "Approved" and f_risk in ("Medium", "High"):
        final_decision = "Review Required"
    else:
        final_decision = loan_decision

    income = float(data.get("income_level", 1) or 1)
    dti = float(data.get("loan_amount_requested", 0)) / income
    confidence = round(abs(loan_prob - 0.5) * 2, 3)

    return {
        "request_id":         data.get("request_id", ""),
        "socket_id":          data.get("socket_id", ""),       # нужен Node.js для routing
        "timestamp":          datetime.utcnow().isoformat(),

        # Кредитное решение
        "loan_decision":      loan_decision,
        "loan_approval_prob": round(loan_prob, 4),
        "rejection_reasons":  get_rejection_reasons(data, loan_prob) if loan_decision == "Rejected" else [],

        # Антифрод
        "fraud_risk":         f_risk,
        "fraud_probability":  round(fraud_prob, 4),
        "fraud_flags":        get_fraud_flags(data),

        # Итог
        "final_decision":     final_decision,
        "confidence_score":   confidence,
        "dti_ratio":          round(dti, 3),
    }


# ─── Kafka helpers ────────────────────────────────────────────────────────────

def create_consumer(retries: int = MAX_RETRIES) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=KAFKA_SERVERS,
                group_id=GROUP_ID,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                key_deserializer=lambda b: b.decode("utf-8") if b else None,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
            )
            log.info("Consumer подключён к %s, топик: %s", KAFKA_SERVERS, INPUT_TOPIC)
            return consumer
        except NoBrokersAvailable:
            log.warning("Попытка %d/%d: Kafka недоступна. Повтор через %ds...",
                        attempt, retries, RETRY_INTERVAL)
            time.sleep(RETRY_INTERVAL)
    raise RuntimeError(f"Не удалось подключиться к Kafka после {retries} попыток")


def create_producer(retries: int = MAX_RETRIES) -> KafkaProducer:
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
            )
            log.info("Producer подключён к %s", KAFKA_SERVERS)
            return producer
        except NoBrokersAvailable:
            log.warning("Попытка %d/%d: Kafka Producer недоступен. Повтор через %ds...",
                        attempt, retries, RETRY_INTERVAL)
            time.sleep(RETRY_INTERVAL)
    raise RuntimeError(f"Не удалось создать Producer после {retries} попыток")


def send(producer: KafkaProducer, topic: str, key: str, payload: dict):
    future = producer.send(topic, key=key, value=payload)
    try:
        future.get(timeout=10)
    except KafkaError as e:
        log.error("Ошибка отправки в топик %s: %s", topic, e)


# ─── Основной цикл ───────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("ML Worker запускается...")
    log.info("  Input  topic : %s", INPUT_TOPIC)
    log.info("  Output topic : %s", OUTPUT_TOPIC)
    log.info("  Logs   topic : %s", LOGS_TOPIC)
    log.info("=" * 60)

    artifacts = load_artifacts()
    consumer  = create_consumer()
    producer  = create_producer()

    log.info("Ожидание сообщений...")

    try:
        for message in consumer:
            start_ts = time.time()
            data = message.value

            request_id = data.get("request_id", f"unknown-{message.offset}")
            log.info("Получен запрос [%s] из топика %s (partition=%d, offset=%d)",
                     request_id, message.topic, message.partition, message.offset)

            try:
                result = build_result(data, artifacts)
                elapsed = round((time.time() - start_ts) * 1000, 1)
                result["processing_ms"] = elapsed

                # → Output: Node.js Consumer читает этот топик и отдаёт клиенту через WebSocket
                send(producer, OUTPUT_TOPIC, key=request_id, payload=result)

                # → Logs: сохраняем полный контекст (input + output) для PostgreSQL
                log_entry = {
                    "request_id":   request_id,
                    "timestamp":    result["timestamp"],
                    "input":        data,
                    "output":       result,
                    "processing_ms": elapsed,
                }
                send(producer, LOGS_TOPIC, key=request_id, payload=log_entry)

                log.info(
                    "[%s] → %s | fraud=%s (%.2f) | loan_prob=%.2f | %dms",
                    request_id,
                    result["final_decision"],
                    result["fraud_risk"],
                    result["fraud_probability"],
                    result["loan_approval_prob"],
                    elapsed,
                )

            except Exception as e:
                log.error("Ошибка обработки [%s]: %s", request_id, e)
                log.debug(traceback.format_exc())

                error_payload = {
                    "request_id":   request_id,
                    "socket_id":    data.get("socket_id", ""),
                    "timestamp":    datetime.utcnow().isoformat(),
                    "error":        str(e),
                    "final_decision": "Error",
                }
                send(producer, OUTPUT_TOPIC, key=request_id, payload=error_payload)

    except KeyboardInterrupt:
        log.info("Остановка по Ctrl+C")
    finally:
        consumer.close()
        producer.flush()
        producer.close()
        log.info("Consumer и Producer закрыты.")


if __name__ == "__main__":
    run()
