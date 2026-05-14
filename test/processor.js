'use strict';

// ─── Вспомогательные функции ──────────────────────────────────────────────────

function rand(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randFloat(min, max, decimals = 1) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

function randItem(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomId(len = 8) {
  return Math.random().toString(36).substring(2, 2 + len);
}

function randomTime(fromHour = 7, toHour = 22) {
  const h = rand(fromHour, toHour);
  const m = rand(0, 59);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

// ─── Генерация пользователя ───────────────────────────────────────────────────

function generateUser(context, events, done) {
  const id = randomId(10);
  context.vars.username  = `user_${id}`;
  context.vars.email     = `user_${id}@test.com`;
  context.vars.password  = `pass_${id}`;
  context.vars.socketId  = `socket_${id}`;
  return done();
}

// ─── Хороший заявитель ────────────────────────────────────────────────────────

function generateGoodApplicant(context, events, done) {
  const occupations   = ['Engineer', 'Doctor', 'Teacher', 'Lawyer', 'Accountant', 'Manager'];
  const maritalStatus = ['Married', 'Single'];
  const purposes      = ['home', 'auto', 'education', 'medical'];
  const devices       = ['Laptop', 'Desktop'];

  context.vars.age               = rand(25, 60);
  context.vars.occupation        = randItem(occupations);
  context.vars.maritalStatus     = randItem(maritalStatus);
  context.vars.dependents        = rand(0, 3);
  context.vars.addressDuration   = rand(12, 120);
  context.vars.creditScore       = rand(680, 850);
  context.vars.incomeLevel       = rand(40000, 150000);
  context.vars.loanAmount        = rand(50000, 400000);
  context.vars.loanTerm          = randItem([12, 24, 36, 48, 60]);
  context.vars.purpose           = randItem(purposes);
  context.vars.interestRate      = randFloat(3.5, 7.0);
  context.vars.previousLoans     = rand(0, 3);
  context.vars.existingLiabilities = rand(0, 10000);
  context.vars.transactionTime   = randomTime(8, 20);
  context.vars.device            = randItem(devices);
  return done();
}

// ─── Рискованный заявитель ────────────────────────────────────────────────────

function generateBadApplicant(context, events, done) {
  const paymentBehaviors = ['Late', 'Defaulted'];
  const nightTimes       = ['01:30', '02:45', '03:15', '04:00', '23:50'];

  context.vars.age               = rand(18, 30);
  context.vars.dependents        = rand(0, 2);
  context.vars.creditScore       = rand(300, 620);
  context.vars.incomeLevel       = rand(5000, 25000);
  context.vars.loanAmount        = rand(100000, 300000);
  context.vars.loanTerm          = randItem([1, 3, 6]);
  context.vars.existingLiabilities = rand(5000, 30000);
  context.vars.paymentBehavior   = randItem(paymentBehaviors);
  context.vars.blacklisted       = Math.random() > 0.5 ? 'Yes' : 'No';
  context.vars.transactionTime   = randItem(nightTimes);
  return done();
}

module.exports = { generateUser, generateGoodApplicant, generateBadApplicant };
