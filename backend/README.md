# CropSight Backend — LLM Summary Proxy

Minimal serverless function that holds the LLM API key server-side and proxies summary requests from the mobile app.

## Setup

1. Install dependencies:
   ```bash
   cd backend
   npm install
   ```

2. Set your LLM API key as an environment variable:
   ```bash
   export LLM_API_KEY=your_api_key_here
   ```

3. Run locally:
   ```bash
   npm start
   # Server runs at http://localhost:8080
   ```

4. Test:
   ```bash
   curl -X POST http://localhost:8080 \
     -H "Content-Type: application/json" \
     -d '{"disease":"leaf_blast","confidence":0.92,"weather":{"temp":28.5,"humidity":85,"condition":"Rain"},"location":{"latitude":15.33,"longitude":119.97}}'
   ```

## Deploy to Google Cloud Functions

```bash
npm run deploy
```

Replace `YOUR_KEY_HERE` in the deploy script with your actual LLM API key, or set it via the GCP console.

## API

**POST /**

Request body:
```json
{
  "disease": "leaf_blast",
  "confidence": 0.92,
  "weather": { "temp": 28.5, "humidity": 85, "condition": "Rain", "wind": 3.2 },
  "location": { "latitude": 15.33, "longitude": 119.97 }
}
```

Response:
```json
{
  "summary": "A leaf blast diagnosis was recorded with 92% AI confidence. At the time of observation, conditions were rainy with 28.5°C temperature and 85% humidity at coordinates 15.33°N, 119.97°E."
}
```

## Notes
- The LLM prompt is explicitly constrained to prevent causal claims
- Weather and location are optional — the summary adapts to available data
- Default LLM endpoint is Gemini 2.0 Flash — change `LLM_ENDPOINT` env var for other providers
