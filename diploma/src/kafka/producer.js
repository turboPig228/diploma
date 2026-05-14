/**
 * kafka/producer.js — Kafka Producer
 * Отправляет запросы на предсказание в топик loan-ml-input
 */

const { Kafka, CompressionTypes, logLevel } = require('kafkajs');

const kafka = new Kafka({
  clientId: process.env.KAFKA_CLIENT_ID || 'loan-dss-server',
  brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
  logLevel: logLevel.WARN,
  retry: {
    initialRetryTime: 300,
    retries: 8,
  },
});

const producer = kafka.producer({
  allowAutoTopicCreation: true,
  transactionTimeout: 30_000,
});

let connected = false;

async function connectProducer() {
  if (connected) return;
  await producer.connect();
  connected = true;
  console.log('[Kafka Producer] Подключён');
}

async function disconnectProducer() {
  if (!connected) return;
  await producer.disconnect();
  connected = false;
  console.log('[Kafka Producer] Отключён');
}

/**
 * Отправляет запрос на предсказание в Python ML-воркер
 * @param {string} requestId  — уникальный ID запроса
 * @param {string} socketId   — ID WebSocket-клиента (для routing ответа)
 * @param {object} inputData  — данные заявителя
 */
async function sendPredictionRequest(requestId, socketId, inputData) {
  const topic = process.env.KAFKA_INPUT_TOPIC || 'loan-ml-input';

  const payload = {
    request_id: requestId,
    socket_id:  socketId,
    ...inputData,
  };

  await producer.send({
    topic,
    compression: CompressionTypes.None,
    messages: [
      {
        key:   requestId,
        value: JSON.stringify(payload),
        headers: { source: 'dss-node', timestamp: Date.now().toString() },
      },
    ],
  });

  console.log(`[Kafka Producer] Запрос отправлен → ${topic} [requestId=${requestId}]`);
  return requestId;
}

module.exports = { connectProducer, disconnectProducer, sendPredictionRequest };
