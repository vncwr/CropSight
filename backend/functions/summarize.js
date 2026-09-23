const https = require('https');

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = 'qwen/qwen3.8-27b';

module.exports = async (req, res) => {
  // Allow CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method === 'GET') {
    return res.status(200).json({
      status: "online",
      message: "CropSight LLM Proxy is active. Accepts { messages }, { prompt }, or { disease, confidence, weather }."
    });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed. Use POST.' });
  }

  try {
    const body = req.body || {};

    // 1. Interactive chat / messages request (e.g. Agronomist bot)
    if (body.messages && Array.isArray(body.messages)) {
      const content = await callGroqMessages(
        body.messages,
        body.max_tokens || 350,
        body.temperature !== undefined ? body.temperature : 0.5
      );
      return res.status(200).json({
        choices: [
          { message: { role: 'assistant', content } }
        ],
        reply: content,
        content: content
      });
    }

    // 2. Custom prompt request (e.g. Forensic analysis or specific summary)
    if (body.prompt) {
      const messages = [
        {
          role: 'system',
          content: body.systemPrompt || 'You are CropSight, an expert plant pathologist and agricultural forensic analyst.'
        },
        { role: 'user', content: body.prompt }
      ];
      const content = await callGroqMessages(
        messages,
        body.max_tokens || 400,
        body.temperature !== undefined ? body.temperature : 0.4
      );
      return res.status(200).json({
        choices: [
          { message: { role: 'assistant', content } }
        ],
        summary: content,
        narrative: content,
        reply: content
      });
    }

    // 3. Structured diagnosis request
    const { disease, confidence, weather } = body;
    if (!disease) {
      return res.status(400).json({ error: 'Missing required field: disease, prompt, or messages' });
    }

    const weatherContext = weather
      ? `Field weather: ${weather.temp}°C, ${weather.humidity}% humidity, condition: ${weather.condition}.`
      : 'Weather data unavailable.';

    const prompt = `A rice farmer photographed a crop diagnosed with ${disease} (${((confidence || 0.9) * 100).toFixed(1)}% confidence). ${weatherContext} Provide 2-3 concise, actionable sentences on what this means and immediate field steps.`;

    const summary = await callGroqMessages([
      {
        role: 'system',
        content: 'You are an agricultural plant pathologist summarizing rice diseases for farmers in simple, practical terms.'
      },
      { role: 'user', content: prompt }
    ], 200, 0.6);

    return res.status(200).json({
      summary,
      choices: [
        { message: { role: 'assistant', content: summary } }
      ]
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};

function callGroqMessages(messages, maxTokens = 350, temperature = 0.6) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      model: GROQ_MODEL,
      messages: messages,
      max_tokens: maxTokens,
      temperature: temperature,
    });

    const request = https.request(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
      },
      (response) => {
        let body = '';
        response.on('data', (chunk) => (body += chunk));
        response.on('end', () => {
          try {
            const data = JSON.parse(body);
            if (data.choices && data.choices[0]) {
              resolve(data.choices[0].message.content.trim());
            } else {
              reject(new Error(body));
            }
          } catch (e) {
            reject(e);
          }
        });
      }
    );

    request.on('error', reject);
    request.write(payload);
    request.end();
  });
}
