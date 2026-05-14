/**
 * routes/auth.js — Регистрация и авторизация
 *
 * POST /api/auth/register  — создать аккаунт
 * POST /api/auth/login     — получить JWT-токен
 */

const router   = require('express').Router();
const bcrypt   = require('bcryptjs');
const jwt      = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');
const { pool } = require('../db');

// ─── Регистрация ──────────────────────────────────────────────────────────────
router.post(
    '/register',
    [
      body('username').trim().isLength({ min: 3, max: 50 }).withMessage('Логин: 3–50 символов'),
      body('email').isEmail().normalizeEmail().withMessage('Некорректный email'),
      body('password').isLength({ min: 6 }).withMessage('Пароль минимум 6 символов'),
    ],
    async (req, res) => {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { username, email, password, role = 'analyst' } = req.body;

      try {
        const exists = await pool.query(
            'SELECT id FROM users WHERE username = $1 OR email = $2',
            [username, email]
        );
        if (exists.rows.length > 0) {
          return res.status(409).json({ error: 'Пользователь уже существует' });
        }

        const hash = await bcrypt.hash(password, 10);
        const result = await pool.query(
            `INSERT INTO users (username, email, password, role)
             VALUES ($1, $2, $3, $4) RETURNING id, username, email, role`,
            [username, email, hash, role]
        );

        res.status(201).json({ message: 'Пользователь создан', user: result.rows[0] });
      } catch (err) {
        console.error('[Auth] Ошибка регистрации:', err.message);
        res.status(500).json({ error: 'Внутренняя ошибка сервера' });
      }
    }
);

// ─── Логин ────────────────────────────────────────────────────────────────────
router.post(
    '/login',
    [
      body('username').trim().notEmpty().withMessage('Укажите логин'),
      body('password').notEmpty().withMessage('Укажите пароль'),
    ],
    async (req, res) => {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { username, password } = req.body;

      try {
        const result = await pool.query(
            'SELECT * FROM users WHERE username = $1',
            [username]
        );
        const user = result.rows[0];

        if (!user || !(await bcrypt.compare(password, user.password))) {
          return res.status(401).json({ error: 'Неверный логин или пароль' });
        }

        const token = jwt.sign(
            { id: user.id, username: user.username, role: user.role },
            process.env.JWT_SECRET || 'secret',
            { expiresIn: process.env.JWT_EXPIRES_IN || '24h' }
        );

        res.json({
          token,
          user: { id: user.id, username: user.username, email: user.email, role: user.role },
        });
      } catch (err) {
        console.error('[Auth] Ошибка логина:', err.message);
        res.status(500).json({ error: 'Внутренняя ошибка сервера' });
      }
    }
);

module.exports = router;

// ─── GET /api/auth/me — проверка токена ──────────────────────────────────────
const authMiddleware = require('../middleware/auth');

router.get('/me', authMiddleware, async (req, res) => {
  try {
    const result = await pool.query(
        'SELECT id, username, email, role FROM users WHERE id = $1',
        [req.user.id]
    );
    if (!result.rows.length) return res.status(404).json({ error: 'Пользователь не найден' });
    res.json({ user: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: 'Ошибка сервера' });
  }
});