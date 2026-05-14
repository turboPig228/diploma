require('dotenv').config({ path: require('path').join(__dirname, '..', 'env') });

const http      = require('http');
const express   = require('express');
const { Server } = require('socket.io');
const { Kafka, logLevel } = require('kafkajs');

const { initDB }              = require('./db');
const { connectProducer, disconnectProducer } = require('./kafka/producer');
const { connectConsumer, disconnectConsumer, setIO } = require('./kafka/consumer');

const authRoutes    = require('./routes/auth');
const predictRoutes = require('./routes/predict');
const historyRoutes = require('./routes/history');

const PORT = process.env.PORT || 3000;

// ─── Создание топиков если не существуют ─────────────────────────────────────
async function ensureTopics() {
  const kafka = new Kafka({
    clientId: 'dss-admin',
    brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
    logLevel: logLevel.WARN,
  });
  const admin = kafka.admin();
  await admin.connect();

  const topics = [
    process.env.KAFKA_INPUT_TOPIC  || 'loan-ml-input',
    process.env.KAFKA_OUTPUT_TOPIC || 'loan-ml-output',
    process.env.KAFKA_LOGS_TOPIC   || 'loan-ml-logs',
  ];

  const existing = await admin.listTopics();
  const toCreate = topics
      .filter(t => !existing.includes(t))
      .map(t => ({ topic: t, numPartitions: 1, replicationFactor: 1 }));

  if (toCreate.length > 0) {
    await admin.createTopics({ topics: toCreate });
    console.log('[Kafka Admin] Топики созданы:', toCreate.map(t => t.topic).join(', '));
  } else {
    console.log('[Kafka Admin] Топики уже существуют:', topics.join(', '));
  }

  await admin.disconnect();
}

// ─── Express ──────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Socket-Id');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// ─── Маршруты ─────────────────────────────────────────────────────────────────
app.use('/api/auth',    authRoutes);
app.use('/api/predict', predictRoutes);
app.use('/api/history', historyRoutes);

app.get("/", (req, res) => {
  res.sendFile(require("path").join(__dirname, "..", "index.html"));
});

app.get("/api/health", (req, res) => {
  res.json({ status: 'ok', service: 'DSS Node.js', timestamp: new Date().toISOString() });
});

// ─── HTTP + Socket.IO ─────────────────────────────────────────────────────────
const server = http.createServer(app);
const io     = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
});

setIO(io);

io.on('connection', (socket) => {
  console.log(`[WS] Клиент подключён: ${socket.id}`);
  socket.emit('connected', { socket_id: socket.id });
  socket.on('disconnect', () => {
    console.log(`[WS] Клиент отключён: ${socket.id}`);
  });
});

// ─── Запуск ───────────────────────────────────────────────────────────────────
async function start() {
  try {
    console.log('[DSS] Инициализация...');

    // 1. БД
    await initDB();

    // 2. Kafka — сначала создаём топики, потом подключаем producer/consumer
    await ensureTopics();
    await connectProducer();
    await connectConsumer();

    // 3. HTTP-сервер
    server.listen(PORT, () => {
      console.log(`\n[DSS] Сервер запущен → http://localhost:${PORT}`);
      console.log('[DSS] Эндпоинты:');
      console.log(`  POST /api/auth/register`);
      console.log(`  POST /api/auth/login`);
      console.log(`  POST /api/predict      (JWT)`);
      console.log(`  GET  /api/history      (JWT)`);
      console.log(`  GET  /api/history/stats (JWT)`);
      console.log(`  GET  /api/health`);
      console.log('[DSS] WebSocket: ws://localhost:' + PORT);
    });
  } catch (err) {
    console.error('[DSS] Ошибка запуска:', err.message);
    process.exit(1);
  }
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────
async function shutdown(signal) {
  console.log(`\n[DSS] Получен ${signal}, завершение...`);
  try {
    await disconnectProducer();
    await disconnectConsumer();
    server.close(() => {
      console.log('[DSS] HTTP-сервер закрыт');
      process.exit(0);
    });
  } catch (err) {
    console.error('[DSS] Ошибка при завершении:', err.message);
    process.exit(1);
  }
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT',  () => shutdown('SIGINT'));

start();