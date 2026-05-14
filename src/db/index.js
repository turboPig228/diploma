/**
 * db/index.js — Подключение к PostgreSQL + инициализация схемы
 */

const { Pool } = require('pg');

const pool = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME     || 'loan_approval',
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  max: 10,
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 5_000,
});

pool.on('error', (err) => {
  console.error('[DB] Неожиданная ошибка пула:', err.message);
});

const INIT_SQL = `
  CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(100) UNIQUE NOT NULL,
    email      VARCHAR(255) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    role       VARCHAR(20)  NOT NULL DEFAULT 'analyst',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    request_id          VARCHAR(64)   NOT NULL UNIQUE,
    user_id             INTEGER       REFERENCES users(id),
    age                 INTEGER,
    occupation          VARCHAR(100),
    marital_status      VARCHAR(50),
    credit_score        INTEGER,
    income_level        NUMERIC(15,2),
    loan_amount         NUMERIC(15,2),
    loan_term           INTEGER,
    purpose_of_loan     VARCHAR(50),
    dti_ratio           NUMERIC(8,3),
    loan_decision       VARCHAR(20),
    loan_approval_prob  NUMERIC(6,4),
    rejection_reasons   JSONB,
    fraud_risk          VARCHAR(20),
    fraud_probability   NUMERIC(6,4),
    fraud_flags         JSONB,
    final_decision      VARCHAR(30),
    confidence_score    NUMERIC(6,3),
    processing_ms       INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  CREATE INDEX IF NOT EXISTS idx_predictions_user      ON predictions(user_id);
  CREATE INDEX IF NOT EXISTS idx_predictions_created   ON predictions(created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_predictions_decision  ON predictions(final_decision);
  CREATE INDEX IF NOT EXISTS idx_predictions_fraud     ON predictions(fraud_risk);
`;

async function initDB() {
  const client = await pool.connect();
  try {
    await client.query(INIT_SQL);
    console.log('[DB] Схема инициализирована');
  } catch (err) {
    console.error('[DB] Ошибка инициализации схемы:', err.message);
    throw err;
  } finally {
    client.release();
  }
}

async function savePrediction(input, output, userId = null) {
  const sql = `
    INSERT INTO predictions (
      request_id, user_id,
      age, occupation, marital_status, credit_score,
      income_level, loan_amount, loan_term, purpose_of_loan, dti_ratio,
      loan_decision, loan_approval_prob, rejection_reasons,
      fraud_risk, fraud_probability, fraud_flags,
      final_decision, confidence_score, processing_ms
    ) VALUES (
               $1,  $2,
               $3,  $4,  $5,  $6,
               $7,  $8,  $9,  $10, $11,
               $12, $13, $14,
               $15, $16, $17,
               $18, $19, $20
             )
      ON CONFLICT (request_id) DO NOTHING
    RETURNING id;
  `;
  const values = [
    output.request_id,
    userId ? parseInt(userId) : null,
    input.age        ? parseInt(input.age)         : null,
    input.occupation || null,
    input.marital_status || null,
    input.credit_score   ? parseInt(input.credit_score)  : null,
    input.income_level   ? parseFloat(input.income_level) : null,
    input.loan_amount_requested ? parseFloat(input.loan_amount_requested) : null,
    input.loan_term      ? parseInt(input.loan_term)      : null,
    input.purpose_of_loan || null,
    output.dti_ratio     ? parseFloat(output.dti_ratio)   : null,
    output.loan_decision  || null,
    output.loan_approval_prob != null ? parseFloat(output.loan_approval_prob) : null,
    JSON.stringify(output.rejection_reasons || []),
    output.fraud_risk     || null,
    output.fraud_probability != null ? parseFloat(output.fraud_probability) : null,
    JSON.stringify(output.fraud_flags || []),
    output.final_decision || null,
    output.confidence_score != null ? parseFloat(output.confidence_score) : null,
    output.processing_ms ? parseInt(output.processing_ms) : null,
  ];
  try {
    const result = await pool.query(sql, values);
    return result.rows[0];
  } catch (err) {
    console.error('[DB] savePrediction error:', err.message);
    throw err;
  }
}

async function getPredictions({ userId, limit = 50, offset = 0, decision, fraudRisk } = {}) {
  let where = [];
  let params = [];
  let i = 1;

  if (userId)    { where.push(`(user_id = $${i++} OR user_id IS NULL)`); params.push(userId); }
  if (decision)  { where.push(`final_decision = $${i++}`); params.push(decision); }
  if (fraudRisk) { where.push(`fraud_risk = $${i++}`);     params.push(fraudRisk); }

  const whereClause = where.length ? `WHERE ${where.join(' AND ')}` : '';
  params.push(limit, offset);

  const sql = `
    SELECT p.*, u.username
    FROM predictions p
           LEFT JOIN users u ON p.user_id = u.id
      ${whereClause}
    ORDER BY p.created_at DESC
      LIMIT $${i++} OFFSET $${i++}
  `;
  const result = await pool.query(sql, params);
  return result.rows;
}

module.exports = { pool, initDB, savePrediction, getPredictions };