const SPAM_PATTERNS = [
  { label: 'URL detected', test: message => /https?:\/\/|www\./i.test(message) },
  { label: 'Phone number', test: message => /\b\d{10,}\b/.test(message.replace(/[\s\-()]/g, '')) },
  { label: '"FREE" keyword', test: message => /\bfree\b/i.test(message) },
  { label: 'Win/Prize word', test: message => /\b(win|won|winner|prize|cash|reward|award)\b/i.test(message) },
  { label: '"CALL" keyword', test: message => /\bcall\b/i.test(message) },
  { label: 'Currency symbol', test: message => /[£$€₹]/.test(message) },
  { label: 'Urgency language', test: message => /\b(now|urgent|expire|expires|limited|tonight|hurry|claim|immediately)\b/i.test(message) },
  { label: 'Excessive exclamation', test: message => (message.match(/!/g) || []).length >= 2 },
];

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Content-Type': 'application/json',
};

exports.handler = async event => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Use POST /predict with a message.' }),
    };
  }

  let payload;
  try {
    payload = JSON.parse(event.body || '{}');
  } catch {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: 'Invalid JSON body.' }),
    };
  }

  const message = String(payload.message || '').trim();
  if (message.length < 1) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: 'Message is required.' }),
    };
  }

  const foundSignals = SPAM_PATTERNS.filter(pattern => pattern.test(message));
  const score = foundSignals.length;
  const hasPhone = SPAM_PATTERNS[1].test(message);

  let spamProbability;
  if (hasPhone && score >= 2) spamProbability = 0.94;
  else if (score >= 4) spamProbability = 0.91;
  else if (score === 3) spamProbability = 0.78;
  else if (score === 2) spamProbability = 0.58;
  else if (score === 1) spamProbability = 0.29;
  else spamProbability = 0.07;

  const isSpam = spamProbability > 0.5;

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify({
      label: isSpam ? 'spam' : 'ham',
      is_spam: isSpam,
      confidence: isSpam ? spamProbability : 1 - spamProbability,
      score,
      found_signals: foundSignals.map(signal => signal.label),
    }),
  };
};
