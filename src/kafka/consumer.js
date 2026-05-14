/**
 * kafka/consumer.js — Kafka Consumer
 *
 * Читает результаты из loan-ml-output:
 *   → находит нужный WebSocket-клиент по socket_id
 *   → отправляет результат клиенту мгновенно
 *
 * Читает логи из loan-ml-logs:
 *   → сохраняет полный контекст в PostgreSQL
 */

const { Kafka, logLevel } = require('kafkajs');
const { savePrediction }  = require('../db');

const kafka = new Kafka({
  clientId: `${process.env.KAFKA_CLIENT_ID || 'loan-dss-server'}-consumer`,
  brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
  logLevel: logLevel.WARN,
  retry: {
    initialRetryTime: 300,
    retries: 8,
  },
});

const consumer = kafka.consumer({
  groupId: process.env.KAFKA_GROUP_ID || 'dss-consumer-group',
  sessionTimeout: 30_000,
  heartbeatInterval: 10_000,
});

let ioRef = null;  // socket.io instance, устанавливается из server.js

function setIO(io) {
  ioRef = io;
}

async function connectConsumer() {
  await consumer.connect();
  console.log('[Kafka Consumer] Подключён');

  const outputTopic = process.env.KAFKA_OUTPUT_TOPIC || 'loan-ml-output';
  const logsTopic   = process.env.KAFKA_LOGS_TOPIC   || 'loan-ml-logs';

  await consumer.subscribe({ topics: [outputTopic, logsTopic], fromBeginning: false });

  await consumer.run({
    eachMessage: async ({ topic, partition, message }) => {
      let payload;
      try {
        payload = JSON.parse(message.value.toString());
      } catch {
        console.error('[Kafka Consumer] Не удалось распарсить сообщение');
        return;
      }

      // ── Топик: результат предсказания → клиент через WebSocket ────────────
      if (topic === outputTopic) {
        const { socket_id, request_id, final_decision } = payload;

        console.log(
            `[Kafka Consumer] Результат получен [${request_id}] → ${final_decision}` +
            ` | socket=${socket_id}`
        );

        if (ioRef && socket_id) {
          // Отправляем конкретному клиенту по его socket_id
          ioRef.to(socket_id).emit('prediction_result', payload);
        } else if (ioRef) {
          // Fallback: broadcast (если socket_id не задан)
          ioRef.emit('prediction_result', payload);
        }
      }

      // ── Топик: полный лог → PostgreSQL ────────────────────────────────────
      if (topic === logsTopic) {
        const { input, output } = payload;
        if (!input || !output) return;

        try {
          const saved = await savePrediction(input, output, input.user_id || null);
          console.log(`[Kafka Consumer] Лог сохранён в БД [${output.request_id}] id=${saved?.id}`);
        } catch (err) {
          console.error('[Kafka Consumer] Ошибка записи в БД:', err.message);
          console.error('[Kafka Consumer] input.age type:', typeof input.age, input.age);
          console.error('[Kafka Consumer] input.credit_score type:', typeof input.credit_score, input.credit_score);
          console.error('[Kafka Consumer] input.loan_term type:', typeof input.loan_term, input.loan_term);
        }
      }
    },
  });
}

async function disconnectConsumer() {
  await consumer.disconnect();
  console.log('[Kafka Consumer] Отключён');
}

module.exports = { connectConsumer, disconnectConsumer, setIO };