# PPG Heart Rate Monitor - Technical Report

## 1. Frontend Components

### 1.1 `page.tsx` (Main Page)
- **Purpose**: Main application page with camera, chart, and controls
- **State**: samples[], heartRate, hrv, signalCombination, inferenceResult, serverResult
- **Flow**: Camera feeds → RGB extraction → PPG computation → Display chart → Valley detection → HR/HRV calculation

### 1.2 `ChartComponent.tsx`
- **Purpose**: Display PPG signal waveform with valley markers
- **Props**: ppgData (number[]), valleys ({index, value}[])
- **Features**: Uses ChartJS Line chart, shows valleys as red points

### 1.3 `SignalCombinationSelector.tsx`
- **Purpose**: Let user choose PPG formula
- **Props**: value, onChange
- **Options**: Default (2R−G−B), Red only, Green only, Blue only, 2×G−R−B

### 1.4 `SimpleCard.tsx`
- **Purpose**: Reusable card for displaying metrics
- **Props**: title, value

### 1.5 `useCamera` Hook
- **Purpose**: Manage camera stream
- **State**: isRecording, error, videoRef, canvasRef
- **Features**: Start/stop camera, draw video to canvas for pixel extraction

### 1.6 `usePPGFromSamples` Hook
- **Purpose**: Process samples and compute heart metrics
- **Output**: valleys[], heartRate {bpm, confidence}, hrv {sdnn, confidence}

---

## 2. Backend API (Flask)

### 2.1 `/health` (GET)
- **Request**: None
- **Response**: `{"ok": true}`
- **Purpose**: Check backend connectivity

### 2.2 `/save-record` (POST)
- **Request**: `{heartRate, hrv, ppgData, timestamp}`
- **Response**: `{success: true, data: record}`
- **Purpose**: Save PPG recording to `records.json`

### 2.3 `/save-labeled-segment` (POST)
- **Request**: `{ppgData: number[], label: "good"|"bad"}`
- **Response**: `{success: true}`
- **Purpose**: Save labeled segment for ML training

### 2.4 `/infer-quality` (POST)
- **Request**: `{ppgData: number[]}`
- **Response**: `{label: "good"|"bad", confidence: number}`
- **Purpose**: Predict signal quality using trained model

### 2.5 `/analyze` (POST)
- **Request**: `{samples: number[], fps: number}`
- **Response**: `{valleys: number[], heartRate: {bpm, confidence}, hrv: {sdnn, confidence}}`
- **Purpose**: Server-side PPG analysis with valley detection

### 2.6 `/upload-model` (POST)
- **Request**: `{model: base64, scaler: base64}`
- **Response**: `{success: true}`
- **Purpose**: Upload trained model files at runtime

### 2.7 `/quality` (POST)
- **Request**: `{samples: number[]}`
- **Response**: `{quality: string, confidence: number}`
- **Purpose**: Simple variance-based quality check

---

## 3. Setup Instructions

### Prerequisites
- Node.js 18+
- Python 3.9+
- Webcam

### Environment Variables
```env
FLASK_URL=http://127.0.0.1:5000
NEXT_PUBLIC_FLASK_URL=http://127.0.0.1:5000
```

### Run Locally

**Backend (Flask)**:
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend (Next.js)**:
```bash
npm install
npm run dev
```

---

## 4. Modifications Made

### 4.1 Signal Formulas (ppg.ts)
Added selectable PPG formulas in `computePPGFromRGB()`:
- `default`: `2*R - G - B`
- `redOnly`: `R`
- `greenOnly`: `G`
- `blueOnly`: `B`
- `2xG-R-B`: `2*G - R - B`

### 4.2 Feature Extraction (ppg_features.py)
Extracts 12 features for ML:
| # | Feature | Description |
|---|---------|-------------|
| 1 | mean | Average signal value |
| 2 | std | Standard deviation |
| 3 | skewness | 3rd moment |
| 4 | kurtosis | 4th moment |
| 5 | signal_range | max - min |
| 6 | zero_crossings | Sign changes |
| 7 | rms | Root mean square |
| 8 | peak_to_peak | Same as range |
| 9 | spectral_energy | FFT energy |
| 10 | dominant_freq | Peak FFT frequency |
| 11 | peaks_count | Local maxima |
| 12 | valleys_count | Local minima |

### 4.3 ML Method
- **Model**: Random Forest Classifier (100 trees, max_depth=10)
- **Features**: 12 statistical + frequency features
- **Training**: Requires labeled data from web UI (good/bad segments)
- **Script**: `train_quality_model.py` - loads labeled_records.json, trains model, saves to .joblib

### 4.4 Layout Changes
- Dark gradient background (slate/purple)
- 2-column grid on desktop (camera+chart left, metrics+controls right)
- Glassmorphism cards with blur effect
- Color-coded metric cards (blue=HR, green=confidence, purple=HRV)
- Added ML training data collection section
- Added model upload section
- Added server-side analysis section
