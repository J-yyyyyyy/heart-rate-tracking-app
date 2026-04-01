# PPG Heart-Rate Monitor

## Prerequisites
Node.js 18+, Python 3.9+, npm or yarn

## Local Setup
### Backend
cd backend && pip install -r requirements.txt && python -m flask --app app run --port 5000

### Frontend
Create .env.local with FLASK_URL=http://127.0.0.1:5000
npm install && npm run dev

Open http://localhost:3000

## Training
1. Collect labeled segments (good/bad)
2. Click "Download labeled_records.json"
3. Save to backend/labeled_records.json
4. Run: cd backend && python train_quality_model.py
5. Upload model and scaler via the app UI

## Deployment
Frontend: Vercel. Backend: PythonAnywhere. See Hosting Guide.

## Links
- Frontend: https://heart-rate-tracking-app.vercel.app
- Backend: https://carolcao.pythonanywhere.com
- GitHub: https://github.com/J-yyyyyyy/heart-rate-tracking-app

## Features
- Real-time PPG signal extraction from camera
- Heart rate and HRV (SDNN) calculation
- 5 signal combination modes: Default (2R−G−B), Red only, Green only, Blue only, 2×G−R−B
- Collect labeled segments (Good/Bad) for ML training
- Download labeled data as JSON
- Train Random Forest classifier with 12 features
- Upload trained model via web UI
- Real-time signal quality inference

## API Endpoints
- GET /api/health - Backend health check
- POST /api/save-record - Save PPG record
- POST /api/save-labeled-segment - Save labeled segment
- POST /api/infer-quality - ML quality inference
- POST /api/upload-model - Upload trained model (base64)
- POST /api/analyze - Server-side valley detection

## Assignment Requirements Checklist
- GitHub repository (public): ✅
- Camera, PPG chart, heart rate, HRV: ✅
- Send labeled segments (good/bad): ✅
- Download JSON button: ✅
- Upload model & scaler buttons: ✅
- 3+ extra signal combinations (5 total): ✅
- Modified layout (different from sample): ✅
- Modified feature extraction (12 features): ✅
- Modified ML method (Random Forest): ✅
- Own trained model (≥10 good, ≥10 bad): ✅
- Deployed on Vercel: ✅
- Deployed on PythonAnywhere: ✅
