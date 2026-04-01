'use client';
import useCamera from './hooks/useCamera';
import SimpleCard from './components/SimpleCard';
import ChartComponent from './components/ChartComponent';
import { useState, useEffect, useRef } from 'react';
import usePPGFromSamples from './hooks/usePPGFromSamples';

import {
  computePPGFromRGB,
  SAMPLES_TO_KEEP,
  MIN_SAMPLES_FOR_DETECTION,
} from './lib/ppg';
import type { SignalCombinationMode } from './components/SignalCombinationSelector';
import SignalCombinationSelector from './components/SignalCombinationSelector';

export default function Home() {
  const { videoRef, canvasRef, isRecording, setIsRecording, error } =
    useCamera();
  const [samples, setSamples] = useState<number[]>([]);
  const [apiResponse, setApiResponse] = useState<object | null>(null);
  const { valleys, heartRate, hrv } = usePPGFromSamples(samples);
  const [signalCombination, setSignalCombination] =
    useState<SignalCombinationMode>('default');

  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  type SegmentLabel = 'good' | 'bad';
  const [segmentLabel, setSegmentLabel] = useState<SegmentLabel>('good');
  const [segmentStatus, setSegmentStatus] = useState<string | null>(null);
  const [labeledSegments, setLabeledSegments] = useState<{ ppgData: number[]; label: string }[]>([]);

  const modelInputRef = useRef<HTMLInputElement>(null);
  const scalerInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  
  const [inferenceResult, setInferenceResult] = useState<{
    label: string | null;
    confidence: number;
    message?: string;
  } | null>(null);

  const samplesRef = useRef<number[]>([]);
  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);

  const INFERENCE_INTERVAL_MS = 2500;
  useEffect(() => {
    if (!isRecording) return;
    let cancelled = false;
    async function run() {
      const current = samplesRef.current;
      if (current.length < MIN_SAMPLES_FOR_DETECTION) return;
      const segment = current.slice(-SAMPLES_TO_KEEP);
      try {
        const res = await fetch('/api/infer-quality', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ppgData: segment }),
        });
        const data = await res.json();
        if (!cancelled) {
          setInferenceResult({
            label: data.label ?? null,
            confidence: data.confidence ?? 0,
            message: data.message ?? data.error ?? undefined,
          });
        }
      } catch {
        if (!cancelled) {
          setInferenceResult({
            label: null,
            confidence: 0,
            message: 'Request failed',
          });
        }
      }
    }
    run();
    const id = setInterval(run, INFERENCE_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isRecording]);

  async function checkBackend() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      setBackendStatus(
        data.ok ? 'Backend OK' : 'Backend returned unexpected data',
      );
    } catch (e) {
      setBackendStatus('Backend unreachable');
    }
  }

  async function sendLabeledSegment() {
    if (samples.length < MIN_SAMPLES_FOR_DETECTION) {
      setSegmentStatus('Need more samples (start recording first)');
      return;
    }
    setSegmentStatus(null);
    const ppgSegment = samples.slice(-SAMPLES_TO_KEEP);
    try {
      const res = await fetch('/api/save-labeled-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ppgData: ppgSegment, label: segmentLabel }),
      });
      const data = await res.json();
      if (data.success) {
        setSegmentStatus(`Saved as ${segmentLabel}`);
        setLabeledSegments((prev) => [...prev, { ppgData: ppgSegment, label: segmentLabel }]);
      } else {
        setSegmentStatus('Error: ' + (data.error || 'Unknown'));
      }
    } catch {
      setSegmentStatus('Error: request failed');
    }
  }

  function downloadLabeledJson() {
    if (labeledSegments.length === 0) return;
    const json = JSON.stringify(labeledSegments, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'labeled_records.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  async function saveRecord() {
    setSaveStatus(null);
    const record = {
      heartRate: { bpm: heartRate.bpm, confidence: heartRate.confidence },
      hrv: {
        sdnn: hrv?.sdnn ?? 0,
        confidence: hrv?.confidence ?? 0,
      },
      ppgData: samples.slice(-SAMPLES_TO_KEEP),
      timestamp: new Date().toISOString(),
    };
    try {
      const res = await fetch('/api/save-record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(record),
      });
      const data = await res.json();
      if (data.success) setSaveStatus('Saved');
      else setSaveStatus('Error: ' + (data.error || 'Unknown'));
    } catch (e) {
      setSaveStatus('Error: request failed');
    }
  }

  async function sendToApi() {
    const res = await fetch('/api/echo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        samples: samples.slice(-10),
        timestamp: Date.now(),
      }),
    });
    const data = await res.json();
    setApiResponse(data);
  }
  
  async function handleUpload() {
    const modelFile = modelInputRef.current?.files?.[0];
    const scalerFile = scalerInputRef.current?.files?.[0];
    if (!modelFile || !scalerFile) {
      setUploadStatus('Select both files');
      return;
    }
    
    // Convert file to base64
    const toBase64 = (file: File) => 
      file.arrayBuffer().then(buf => btoa(String.fromCharCode(...new Uint8Array(buf))));
    
    try {
      const model = await toBase64(modelFile);
      const scaler = await toBase64(scalerFile);
      const res = await fetch('/api/upload-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, scaler }),
      });
      const data = await res.json();
      setUploadStatus(data.success ? 'Uploaded!' : data.error);
    } catch {
      setUploadStatus('Upload failed');
    }
  }
    // 在 Home 组件内部添加这些状态
  const [serverResult, setServerResult] = useState<{
    heartRate: { bpm: number; confidence: number };
    hrv: { sdnn: number; confidence: number };
    valleys?: number[];
    loading?: boolean;
    error?: string;
  } | null>(null);

  // 添加服务器分析函数
  async function analyzeOnServer() {
    if (samples.length < MIN_SAMPLES_FOR_DETECTION) {
      setServerResult({
        heartRate: { bpm: 0, confidence: 0 },
        hrv: { sdnn: 0, confidence: 0 },
        error: `Need at least ${MIN_SAMPLES_FOR_DETECTION} samples`
      });
      return;
    }

    setServerResult(prev => prev ? { ...prev, loading: true, error: undefined } : {
      heartRate: { bpm: 0, confidence: 0 },
      hrv: { sdnn: 0, confidence: 0 },
      loading: true
    });

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples: samples.slice(-SAMPLES_TO_KEEP),
          fps: 30, // 你的采样帧率，根据实际情况调整
        }),
      });
      
      const data = await res.json();
      
      if (data.error) {
        setServerResult({
          heartRate: { bpm: 0, confidence: 0 },
          hrv: { sdnn: 0, confidence: 0 },
          error: data.error
        });
      } else {
        setServerResult({
          heartRate: data.heartRate,
          hrv: data.hrv,
          valleys: data.valleys,
          loading: false
        });
      }
    } catch (error) {
      setServerResult({
        heartRate: { bpm: 0, confidence: 0 },
        hrv: { sdnn: 0, confidence: 0 },
        error: 'Request failed - is Flask running?'
      });
    }
  }

  useEffect(() => {
    const video = videoRef.current;
    const c = canvasRef.current;
    if (!isRecording || !video || !c) return;

    const ctx = c.getContext('2d');
    if (!ctx) return;

    let running = true;
    function tick() {
      if (!running || !ctx) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v?.srcObject || !v.videoWidth || !c) {
        requestAnimationFrame(tick);
        return;
      }
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      ctx.drawImage(v, 0, 0);
      const w = 10,
        h = 10;
      const x = (c.width - w) / 2;
      const y = (c.height - h) / 2;
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const data = ctx.getImageData(x, y, w, h).data;
      let rSum = 0,
        gSum = 0,
        bSum = 0,
        pixelCount = 0;
      for (let i = 0; i < data.length; i += 4) {
        rSum += data[i];
        gSum += data[i + 1];
        bSum += data[i + 2];
        pixelCount += 1;
      }
      const ppgValue = computePPGFromRGB(
        rSum,
        gSum,
        bSum,
        pixelCount,
        signalCombination,
      );

      setSamples((prev) => [...prev.slice(-(SAMPLES_TO_KEEP - 1)), ppgValue]);

      requestAnimationFrame(tick);
    }
    tick();
    return () => {
      running = false;
    };
  }, [isRecording, signalCombination]);

  
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">PPG Heart Rate Monitor</h1>
          <p className="text-purple-200">Real-time photoplethysmography analysis</p>
        </header>

        {/* Main Grid: 2 columns on desktop, 1 on mobile */}
        <div className="grid lg:grid-cols-2 gap-6">
          
          {/* LEFT COLUMN: Camera + Chart */}
          <div className="space-y-6">
            {/* Camera Card */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-3">Camera</h2>
              <div className="bg-black rounded-xl overflow-hidden aspect-video flex items-center justify-center">
                <video ref={videoRef} autoPlay muted playsInline className="hidden" />
                {isRecording ? (
                  <canvas ref={canvasRef} className="w-full h-full object-contain" />
                ) : (
                  <span className="text-gray-500 text-sm">Start recording to see camera</span>
                )}
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setIsRecording((r) => !r)}
                  className={`px-6 py-2 rounded-full font-semibold transition ${
                    isRecording 
                      ? 'bg-red-500 hover:bg-red-600 text-white' 
                      : 'bg-green-500 hover:bg-green-600 text-white'
                  }`}
                >
                  {isRecording ? '⏹ Stop Recording' : '▶ Start Recording'}
                </button>
                {error && <p className="text-red-400 mt-2 text-sm">{error}</p>}
              </div>
            </div>

            {/* Chart Card */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-xl font-semibold text-white">PPG Signal</h2>
                <SignalCombinationSelector
                  value={signalCombination}
                  onChange={setSignalCombination}
                />
              </div>
              <ChartComponent
                ppgData={samples.slice(-SAMPLES_TO_KEEP)}
                valleys={valleys}
              />
            </div>
          </div>

          {/* RIGHT COLUMN: Metrics + Controls */}
          <div className="space-y-6">
            {/* Metrics Cards Row */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gradient-to-br from-blue-500 to-blue-700 rounded-xl p-4 text-white text-center">
                <p className="text-sm opacity-80">Heart Rate</p>
                <p className="text-3xl font-bold">{heartRate.bpm > 0 ? `${heartRate.bpm}` : '--'}</p>
                <p className="text-xs opacity-75">bpm</p>
              </div>
              <div className="bg-gradient-to-br from-green-500 to-green-700 rounded-xl p-4 text-white text-center">
                <p className="text-sm opacity-80">Confidence</p>
                <p className="text-3xl font-bold">{heartRate.confidence > 0 ? `${(heartRate.confidence * 100).toFixed(0)}` : '--'}</p>
                <p className="text-xs opacity-75">%</p>
              </div>
              <div className="bg-gradient-to-br from-purple-500 to-purple-700 rounded-xl p-4 text-white text-center">
                <p className="text-sm opacity-80">HRV (SDNN)</p>
                <p className="text-3xl font-bold">{hrv.sdnn > 0 ? `${hrv.sdnn}` : '--'}</p>
                <p className="text-xs opacity-75">ms</p>
              </div>
            </div>

            {/* Live Data Card */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">Live Data</h2>
              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-purple-200">Current PPG</p>
                  <p className="text-2xl font-mono text-white">{samples[samples.length - 1]?.toFixed(1) ?? '-'}</p>
                </div>
                <div>
                  <p className="text-sm text-purple-200">Last 20</p>
                  <p className="text-sm font-mono text-white max-w-[200px] truncate">
                    {samples.slice(-20).map((s) => s.toFixed(0)).join(', ') || '-'}
                  </p>
                </div>
              </div>
            </div>

            {/* Actions Card */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">Actions</h2>
              <div className="flex gap-3">
                <button onClick={checkBackend} className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition">
                  Check Backend
                </button>
                <button onClick={saveRecord} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition">
                  Save Record
                </button>
              </div>
              {backendStatus && <p className="mt-2 text-sm text-purple-200">{backendStatus}</p>}
              {saveStatus && <p className="mt-2 text-sm text-purple-200">{saveStatus}</p>}
            </div>

            {/* ML Section: Collect Labeled Data */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">📝 Collect Training Data</h2>
              <p className="text-sm text-purple-200 mb-3">Label the current PPG segment for ML training</p>
              
              <div className="flex items-center gap-4 mb-3">
                <label className="flex items-center gap-2 text-white">
                  <input type="radio" name="segmentLabel" checked={segmentLabel === 'good'} onChange={() => setSegmentLabel('good')} />
                  <span className="text-green-400">✓ Good</span>
                </label>
                <label className="flex items-center gap-2 text-white">
                  <input type="radio" name="segmentLabel" checked={segmentLabel === 'bad'} onChange={() => setSegmentLabel('bad')} />
                  <span className="text-red-400">✗ Bad</span>
                </label>
              </div>
              
              <div className="flex gap-2">
                <button onClick={sendLabeledSegment} className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg transition">
                  Save Segment
                </button>
                <button onClick={downloadLabeledJson} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition disabled:opacity-50" disabled={labeledSegments.length === 0}>
                  Download ({labeledSegments.length})
                </button>
              </div>
              {segmentStatus && <p className="mt-2 text-sm text-purple-200">{segmentStatus}</p>}
            </div>

            {/* ML Section: Signal Quality Inference */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">🤖 Quality Inference</h2>
              <div className="text-sm">
                {inferenceResult?.message && <p className="text-purple-200">{inferenceResult.message}</p>}
                {inferenceResult?.label ? (
                  <p className="text-white">
                    Predicted: <strong className={inferenceResult.label === 'good' ? 'text-green-400' : 'text-red-400'}>{inferenceResult.label}</strong>
                    {inferenceResult.confidence > 0 && ` (${(inferenceResult.confidence * 100).toFixed(0)}% confidence)`}
                  </p>
                ) : (
                  <p className="text-purple-200">
                    {isRecording && samples.length < MIN_SAMPLES_FOR_DETECTION ? 'Collecting samples…' :
                    !isRecording ? 'Start recording' : '--'}
                  </p>
                )}
              </div>
            </div>

            {/* ML Section: Upload Model */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">📤 Upload Model</h2>
              <div className="mb-2">
                <p className="text-sm text-purple-200">Model file (.joblib):</p>
                <input type="file" ref={modelInputRef} accept=".joblib" className="text-sm text-white file:mr-2 file:py-1 file:px-3 file:rounded-lg file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700" />
              </div>
              <div className="mb-3">
                <p className="text-sm text-purple-200">Scaler file (.joblib):</p>
                <input type="file" ref={scalerInputRef} accept=".joblib" className="text-sm text-white file:mr-2 file:py-1 file:px-3 file:rounded-lg file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700" />
              </div>
              <button onClick={handleUpload} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition">
                Upload Model & Scaler
              </button>
              {uploadStatus && <p className="mt-2 text-sm text-purple-200">{uploadStatus}</p>}
            </div>

            {/* Server-side Analysis */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20">
              <h2 className="text-lg font-semibold text-white mb-2">🖥 Server Analysis</h2>
              <button onClick={analyzeOnServer} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition" disabled={serverResult?.loading}>
                {serverResult?.loading ? 'Analyzing...' : 'Run Server Analysis'}
              </button>
              
              {serverResult && (
                <div className="mt-3 p-3 bg-black/30 rounded-lg">
                  {serverResult.error ? (
                    <p className="text-red-400 text-sm">{serverResult.error}</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-3 text-center">
                        <div>
                          <p className="text-xs text-purple-200">Server HR</p>
                          <p className="text-xl font-bold text-white">{serverResult.heartRate.bpm > 0 ? `${serverResult.heartRate.bpm} bpm` : '--'}</p>
                        </div>
                        <div>
                          <p className="text-xs text-purple-200">Server HRV</p>
                          <p className="text-xl font-bold text-white">{serverResult.hrv.sdnn > 0 ? `${serverResult.hrv.sdnn} ms` : '--'}</p>
                        </div>
                      </div>
                      {heartRate.bpm > 0 && serverResult.heartRate.bpm > 0 && (
                        <div className="mt-2 pt-2 border-t border-white/20 text-center">
                          <p className="text-xs text-purple-200">
                            Difference: {Math.abs(heartRate.bpm - serverResult.heartRate.bpm).toFixed(1)} bpm
                          </p>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
