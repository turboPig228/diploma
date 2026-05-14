/**
 * routes/history.js — История предсказаний
 *
 * GET /api/history               — список (с пагинацией и фильтрами)
 * GET /api/history/:request_id   — детали конкретного предсказания
 * GET /api/history/stats         — сводная статистика
 */

const router = require('express').Router();
const auth   = require('../middleware/auth');
const { pool, getPredictions } = require('../db');

// ─── GET /api/history ─────────────────────────────────────────────────────────
router.get('/', auth, async (req, res) => {
  if (!req.user) return res.status(401).json({ error: 'Не авторизован' });
  try {
    const limit     = Math.min(parseInt(req.query.limit  || '50'), 200);
    const offset    = parseInt(req.query.offset || '0');
    const decision  = req.query.decision  || null;  // Approved / Rejected / Review Required
    const fraudRisk = req.query.fraud_risk || null;  // Low / Medium / High

    // Admin видит все, аналитик видит свои + записи без user_id
    const userId = req.user.role === 'admin' ? null : req.user.id;

    const rows = await getPredictions({ userId, limit, offset, decision, fraudRisk });

    res.json({
      total:  rows.length,
      limit,
      offset,
      items:  rows,
    });
  } catch (err) {
    console.error('[History] Ошибка:', err.message);
    res.status(500).json({ error: 'Не удалось получить историю' });
  }
});

// ─── GET /api/history/stats ───────────────────────────────────────────────────
router.get('/stats', auth, async (req, res) => {
  if (!req.user) return res.status(401).json({ error: 'Не авторизован' });
  try {
    const userId = req.user.role === 'admin' ? null : req.user.id;
    const userFilter = userId ? 'WHERE p.user_id = $1' : '';
    const params     = userId ? [userId] : [];

    const sql = `
      SELECT
        COUNT(*)                                              AS total,
        COUNT(*) FILTER (WHERE final_decision = 'Approved')  AS approved,
          COUNT(*) FILTER (WHERE final_decision = 'Rejected')  AS rejected,
          COUNT(*) FILTER (WHERE final_decision = 'Review Required') AS review,
          COUNT(*) FILTER (WHERE fraud_risk = 'High')           AS high_fraud,
          COUNT(*) FILTER (WHERE fraud_risk = 'Medium')         AS medium_fraud,
          COUNT(*) FILTER (WHERE fraud_risk = 'Low')            AS low_fraud,
          ROUND(AVG(loan_approval_prob)::numeric, 4)            AS avg_approval_prob,
        ROUND(AVG(fraud_probability)::numeric, 4)             AS avg_fraud_prob,
        ROUND(AVG(credit_score)::numeric, 1)                  AS avg_credit_score,
        ROUND(AVG(dti_ratio)::numeric, 3)                     AS avg_dti,
        ROUND(AVG(processing_ms)::numeric, 1)                 AS avg_processing_ms
      FROM predictions p
        ${userFilter}
    `;

    const result = await pool.query(sql, params);
    res.json(result.rows[0]);
  } catch (err) {
    console.error('[History/Stats] Ошибка:', err.message);
    res.status(500).json({ error: 'Не удалось получить статистику' });
  }
});

// ─── GET /api/history/:request_id ────────────────────────────────────────────
router.get('/:request_id', auth, async (req, res) => {
  if (!req.user) return res.status(401).json({ error: 'Не авторизован' });
  try {
    const { request_id } = req.params;
    const result = await pool.query(
        `SELECT p.*, u.username
         FROM predictions p
                LEFT JOIN users u ON p.user_id = u.id
         WHERE p.request_id = $1`,
        [request_id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Предсказание не найдено' });
    }

    const row = result.rows[0];

    // Аналитик не может смотреть чужие записи
    if (req.user.role !== 'admin' && row.user_id !== req.user.id) {
      return res.status(403).json({ error: 'Нет доступа' });
    }

    res.json(row);
  } catch (err) {
    console.error('[History/:id] Ошибка:', err.message);
    res.status(500).json({ error: 'Не удалось получить запись' });
  }
});

module.exports = router;