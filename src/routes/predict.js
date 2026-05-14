/**
 * routes/predict.js — Отправка заявки на предсказание
 *
 * POST /api/predict  — валидация + отправка в Kafka → ответ придёт через WebSocket
 */

const router  = require('express').Router();
const { v4: uuidv4 } = require('uuid');
const { body, validationResult } = require('express-validator');
const auth    = require('../middleware/auth');
const { sendPredictionRequest } = require('../kafka/producer');

// ─── Валидация всех 28 полей ──────────────────────────────────────────────────
const validatePrediction = [
  // Демографические
  body('age').isInt({ min: 18, max: 100 }).withMessage('Возраст: 18–100'),
  body('occupation').trim().notEmpty().withMessage('Профессия обязательна'),
  body('marital_status')
    .isIn(['Single', 'Married', 'Divorced', 'Widowed'])
    .withMessage('Семейное положение: Single / Married / Divorced / Widowed'),
  body('dependents').isInt({ min: 0, max: 20 }).withMessage('Иждивенцы: 0–20'),
  body('residential_status')
    .isIn(['Own', 'Rent', 'Live with Parents'])
    .withMessage('Статус жилья: Own / Rent / Live with Parents'),
  body('address_duration').isInt({ min: 0 }).withMessage('Срок проживания ≥ 0'),

  // Финансовые
  body('credit_score').isInt({ min: 300, max: 850 }).withMessage('Кредитный рейтинг: 300–850'),
  body('income_level').isFloat({ min: 1 }).withMessage('Доход должен быть > 0'),
  body('loan_amount_requested').isFloat({ min: 1 }).withMessage('Сумма кредита > 0'),
  body('loan_term').isInt({ min: 1, max: 360 }).withMessage('Срок кредита: 1–360 мес.'),
  body('purpose_of_loan')
    .isIn(['home', 'auto', 'personal', 'education', 'travel', 'medical', 'business'])
    .withMessage('Цель кредита не из допустимых значений'),
  body('collateral').isIn(['Yes', 'No']).withMessage('Залог: Yes / No'),
  body('interest_rate').isFloat({ min: 0, max: 100 }).withMessage('Ставка: 0–100'),
  body('previous_loans').isInt({ min: 0 }).withMessage('Прошлые кредиты ≥ 0'),
  body('existing_liabilities').isFloat({ min: 0 }).withMessage('Обязательства ≥ 0'),

  // Поведенческие
  body('application_behavior')
    .isIn(['Normal', 'Rapid'])
    .withMessage('Поведение при подаче: Normal / Rapid'),
  body('location_of_application')
    .isIn(['Local', 'Unusual'])
    .withMessage('Местоположение: Local / Unusual'),
  body('change_in_behavior')
    .isIn(['Yes', 'No'])
    .withMessage('Изменение поведения: Yes / No'),
  body('time_of_transaction')
    .matches(/^\d{1,2}:\d{2}$/)
    .withMessage('Время транзакции в формате HH:MM'),
  body('account_activity')
    .isIn(['Normal', 'Unusual'])
    .withMessage('Активность счёта: Normal / Unusual'),
  body('payment_behavior')
    .isIn(['On-time', 'Late', 'Defaulted'])
    .withMessage('История платежей: On-time / Late / Defaulted'),

  // Верификация
  body('blacklists').isIn(['Yes', 'No']).withMessage('Чёрный список: Yes / No'),
  body('employment_verification')
    .isIn(['Verified', 'Not Verified'])
    .withMessage('Верификация: Verified / Not Verified'),
  body('past_financial_malpractices')
    .isIn(['Yes', 'No'])
    .withMessage('Финансовые нарушения: Yes / No'),

  // Технические
  body('device_information')
    .isIn(['Mobile', 'Laptop', 'Desktop', 'Tablet'])
    .withMessage('Устройство: Mobile / Laptop / Desktop / Tablet'),
  body('social_media_footprint')
    .isIn(['Yes', 'No'])
    .withMessage('Соцсети: Yes / No'),
  body('consistency_in_data')
    .isIn(['Consistent', 'Inconsistent'])
    .withMessage('Согласованность данных: Consistent / Inconsistent'),
  body('referral')
    .isIn(['Referral', 'Online'])
    .withMessage('Источник: Referral / Online'),
];

// ─── POST /api/predict ────────────────────────────────────────────────────────
router.post('/', auth, validatePrediction, async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    const requestId = `REQ-${uuidv4()}`;
    const socketId  = req.body.socket_id || req.headers['x-socket-id'] || '';

    // Убираем socket_id из входных данных перед отправкой
    const { socket_id, ...inputData } = req.body;

    await sendPredictionRequest(requestId, socketId, inputData);

    // Сразу отвечаем клиенту — результат придёт через WebSocket
    res.status(202).json({
      message:    'Запрос принят. Результат придёт через WebSocket.',
      request_id: requestId,
    });
  } catch (err) {
    console.error('[Predict] Ошибка:', err.message);
    res.status(500).json({ error: 'Не удалось отправить запрос в очередь' });
  }
});

module.exports = router;
