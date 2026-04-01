'use client';

import { useEffect, useRef, useState } from 'react';
import useCamera from './hooks/useCamera';
import usePPGFromSamples from './hooks/usePPGFromSamples';
import ChartComponent from './components/ChartComponent';
import SignalCombinationSelector from './components/SignalCombinationSelector';
import type { SignalCombinationMode } from './components/SignalCombinationSelector';
import {
  computePPGFromRGB,
  SAMPLES_TO_KEEP,
  MIN_SAMPLES_FOR_DETECTION,
} from './lib/ppg';

type SegmentLabel = 'good' | 'bad';

type ServerResult = {
  heartRate: { bpm: number; confidence: number };
  hrv: { sdnn: number; confidence: number };
  valleys?: number[];
  loading?: boolean;
  error?: string;
};

export default function Home() {
  const { videoRef, canvasRef, isRecording, setIsRecording, error } =
    useCamera();

  const [samples, setSamples] = useState<number[]>([]);
  const { valleys, heartRate, hrv } = usePPGFromSamples(samples);

  const [signalCombination, setSignalCombination] =
    useState<SignalCombinationMode>('default');

  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const [segmentLabel, setSegmentLabel] = useState<SegmentLabel>('good');
  const [segmentStatus, setSegmentStatus] = useState<string | null>(null);
  const [labeledSegments, setLabeledSegments] = useState<
    { ppgData: number[]; label: string }[]
  >([]);

  const modelInputRef = useRef<HTMLInputElement>(null);
  const scalerInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const [inferenceResult, setInferenceResult] = useState<{
    label: string | null;
    confidence: number;
    message?: string;
  } | null>(null);

  const [serverResult, setServerResult] = useState<ServerResult | null>(null);

  const samplesRef = useRef<number[]>([]);

  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);

  useEffect(() => {
    if (!isRecording) return;

    const INFERENCE_INTERVAL_MS = 2500;
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
    } catch {
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
        body: JSON.stringify({
          ppgData: ppgSegment,
          label: segmentLabel,
        }),
      });

      const data = await res.json();

      if (data.success) {
        setSegmentStatus(`Saved as ${segmentLabel}`);
        setLabeledSegments((prev) => [
          ...prev,
          { ppgData: ppgSegment, label: segmentLabel },
        ]);
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
      heartRate: {
        bpm: heartRate.bpm,
        confidence: heartRate.confidence,
      },
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

      if (data.success) {
        setSaveStatus('Saved');
      } else {
        setSaveStatus('Error: ' + (data.error || 'Unknown'));
      }
    } catch {
      setSaveStatus('Error: request failed');
    }
  }

  async function handleUpload() {
    const modelFile = modelInputRef.current?.files?.[0];
    const scalerFile = scalerInputRef.current?.files?.[0];

    if (!modelFile || !scalerFile) {
      setUploadStatus('Select both files');
      return;
    }

    const toBase64 = async (file: File) => {
      const buffer = await file.arrayBuffer();
      return btoa(String.fromCharCode(...new Uint8Array(buffer)));
    };

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

  async function analyzeOnServer() {
    if (samples.length < MIN_SAMPLES_FOR_DETECTION) {
      setServerResult({
        heartRate: { bpm: 0, confidence: 0 },
        hrv: { sdnn: 0, confidence: 0 },
        error: `Need at least ${MIN_SAMPLES_FOR_DETECTION} samples`,
      });
      return;
    }

    setServerResult((prev) =>
      prev
        ? { ...prev, loading: true, error: undefined }
        : {
            heartRate: { bpm: 0, confidence: 0 },
            hrv: { sdnn: 0, confidence: 0 },
            loading: true,
          },
    );

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples: samples.slice(-SAMPLES_TO_KEEP),
          fps: 30,
        }),
      });

      const data = await res.json();

      if (data.error) {
        setServerResult({
          heartRate: { bpm: 0, confidence: 0 },
          hrv: { sdnn: 0, confidence: 0 },
          error: data.error,
          loading: false,
        });
      } else {
        setServerResult({
          heartRate: data.heartRate,
          hrv: data.hrv,
          valleys: data.valleys,
          loading: false,
        });
      }
    } catch {
      setServerResult({
        heartRate: { bpm: 0, confidence: 0 },
        hrv: { sdnn: 0, confidence: 0 },
        error: 'Request failed - is Flask running?',
        loading: false,
      });
    }
  }

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!isRecording || !video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let running = true;

    function tick() {
      if (!running) return;

      const v = videoRef.current;
      const c = canvasRef.current;

      if (!v?.srcObject || !v.videoWidth || !c) {
        requestAnimationFrame(tick);
        return;
      }

      c.width = v.videoWidth;
      c.height = v.videoHeight;

      ctx.drawImage(v, 0, 0);

      const w = 10;
      const h = 10;
      const x = (c.width - w) / 2;
      const y = (c.height - h) / 2;

      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const image = ctx.getImageData(x, y, w, h).data;

      let rSum = 0;
      let gSum = 0;
      let bSum = 0;
      let pixelCount = 0;

      for (let i = 0; i < image.length; i += 4) {
        rSum += image[i];
        gSum += image[i + 1];
        bSum += image[i + 2];
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
  }, [isRecording, signalCombination, videoRef, canvasRef]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 text-center">
          <h1 className="mb-2 text-4xl font-bold text-white">
            PPG Heart Rate Monitor
          </h1>
          <p className="text-purple-200">
            Real-time photoplethysmography analysis
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-3 text-xl font-semibold text-white">Camera</h2>

              <div className="aspect-video overflow-hidden rounded-xl bg-black flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="hidden"
                />

                {isRecording ? (
                  <canvas
                    ref={canvasRef}
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <span className="text-sm text-gray-500">
                    Start recording to see camera
                  </span>
                )}
              </div>

              <div className="mt-4">
                <button
                  onClick={() => setIsRecording((r) => !r)}
                  className={`rounded-full px-6 py-2 font-semibold transition ${
                    isRecording
                      ? 'bg-red-500 text-white hover:bg-red-600'
                      : 'bg-green-500 text-white hover:bg-green-600'
                  }`}
                >
                  {isRecording ? '⏹ Stop Recording' : '▶ Start Recording'}
                </button>

                {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
              </div>
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <div className="mb-3 flex items-center justify-between">
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

          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 p-4 text-center text-white">
                <p className="text-sm opacity-80">Heart Rate</p>
                <p className="text-3xl font-bold">
                  {heartRate.bpm > 0 ? `${heartRate.bpm}` : '--'}
                </p>
                <p className="text-xs opacity-75">bpm</p>
              </div>

              <div className="rounded-xl bg-gradient-to-br from-green-500 to-green-700 p-4 text-center text-white">
                <p className="text-sm opacity-80">Confidence</p>
                <p className="text-3xl font-bold">
                  {heartRate.confidence > 0
                    ? `${(heartRate.confidence * 100).toFixed(0)}`
                    : '--'}
                </p>
                <p className="text-xs opacity-75">%</p>
              </div>

              <div className="rounded-xl bg-gradient-to-br from-purple-500 to-purple-700 p-4 text-center text-white">
                <p className="text-sm opacity-80">HRV (SDNN)</p>
                <p className="text-3xl font-bold">
                  {(hrv?.sdnn ?? 0) > 0 ? `${hrv?.sdnn}` : '--'}
                </p>
                <p className="text-xs opacity-75">ms</p>
              </div>
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">
                Live Data
              </h2>

              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-purple-200">Current PPG</p>
                  <p className="text-2xl font-mono text-white">
                    {samples[samples.length - 1]?.toFixed(1) ?? '-'}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-purple-200">Last 20</p>
                  <p className="max-w-[200px] truncate text-sm font-mono text-white">
                    {samples
                      .slice(-20)
                      .map((s) => s.toFixed(0))
                      .join(', ') || '-'}
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">Actions</h2>

              <div className="flex gap-3">
                <button
                  onClick={checkBackend}
                  className="rounded-lg bg-gray-600 px-4 py-2 text-white transition hover:bg-gray-700"
                >
                  Check Backend
                </button>

                <button
                  onClick={saveRecord}
                  className="rounded-lg bg-emerald-600 px-4 py-2 text-white transition hover:bg-emerald-700"
                >
                  Save Record
                </button>
              </div>

              {backendStatus && (
                <p className="mt-2 text-sm text-purple-200">{backendStatus}</p>
              )}
              {saveStatus && (
                <p className="mt-2 text-sm text-purple-200">{saveStatus}</p>
              )}
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">
                📝 Collect Training Data
              </h2>
              <p className="mb-3 text-sm text-purple-200">
                Label the current PPG segment for ML training
              </p>

              <div className="mb-3 flex items-center gap-4">
                <label className="flex items-center gap-2 text-white">
                  <input
                    type="radio"
                    name="segmentLabel"
                    checked={segmentLabel === 'good'}
                    onChange={() => setSegmentLabel('good')}
                  />
                  <span className="text-green-400">✓ Good</span>
                </label>

                <label className="flex items-center gap-2 text-white">
                  <input
                    type="radio"
                    name="segmentLabel"
                    checked={segmentLabel === 'bad'}
                    onChange={() => setSegmentLabel('bad')}
                  />
                  <span className="text-red-400">✗ Bad</span>
                </label>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={sendLabeledSegment}
                  className="rounded-lg bg-amber-600 px-4 py-2 text-white transition hover:bg-amber-700"
                >
                  Save Segment
                </button>

                <button
                  onClick={downloadLabeledJson}
                  disabled={labeledSegments.length === 0}
                  className="rounded-lg bg-green-600 px-4 py-2 text-white transition hover:bg-green-700 disabled:opacity-50"
                >
                  Download ({labeledSegments.length})
                </button>
              </div>

              {segmentStatus && (
                <p className="mt-2 text-sm text-purple-200">{segmentStatus}</p>
              )}
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">
                🤖 Quality Inference
              </h2>

              <div className="text-sm">
                {inferenceResult?.message && (
                  <p className="text-purple-200">{inferenceResult.message}</p>
                )}

                {inferenceResult?.label ? (
                  <p className="text-white">
                    Predicted:{' '}
                    <strong
                      className={
                        inferenceResult.label === 'good'
                          ? 'text-green-400'
                          : 'text-red-400'
                      }
                    >
                      {inferenceResult.label}
                    </strong>
                    {inferenceResult.confidence > 0 &&
                      ` (${(inferenceResult.confidence * 100).toFixed(0)}% confidence)`}
                  </p>
                ) : (
                  <p className="text-purple-200">
                    {isRecording &&
                    samples.length < MIN_SAMPLES_FOR_DETECTION
                      ? 'Collecting samples…'
                      : !isRecording
                        ? 'Start recording'
                        : '--'}
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">
                📤 Upload Model
              </h2>

              <div className="mb-2">
                <p className="text-sm text-purple-200">Model file (.joblib):</p>
                <input
                  type="file"
                  ref={modelInputRef}
                  accept=".joblib"
                  className="text-sm text-white file:mr-2 file:rounded-lg file:border-0 file:bg-purple-600 file:px-3 file:py-1 file:text-white hover:file:bg-purple-700"
                />
              </div>

              <div className="mb-3">
                <p className="text-sm text-purple-200">
                  Scaler file (.joblib):
                </p>
                <input
                  type="file"
                  ref={scalerInputRef}
                  accept=".joblib"
                  className="text-sm text-white file:mr-2 file:rounded-lg file:border-0 file:bg-purple-600 file:px-3 file:py-1 file:text-white hover:file:bg-purple-700"
                />
              </div>

              <button
                onClick={handleUpload}
                className="rounded-lg bg-blue-600 px-4 py-2 text-white transition hover:bg-blue-700"
              >
                Upload Model & Scaler
              </button>

              {uploadStatus && (
                <p className="mt-2 text-sm text-purple-200">{uploadStatus}</p>
              )}
            </div>

            <div className="rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-lg">
              <h2 className="mb-2 text-lg font-semibold text-white">
                🖥 Server Analysis
              </h2>

              <button
                onClick={analyzeOnServer}
                disabled={serverResult?.loading}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-white transition hover:bg-indigo-700 disabled:opacity-50"
              >
                {serverResult?.loading ? 'Analyzing...' : 'Run Server Analysis'}
              </button>

              {serverResult && (
                <div className="mt-3 rounded-lg bg-black/30 p-3">
                  {serverResult.error ? (
                    <p className="text-sm text-red-400">{serverResult.error}</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-3 text-center">
                        <div>
                          <p className="text-xs text-purple-200">Server HR</p>
                          <p className="text-xl font-bold text-white">
                            {serverResult.heartRate.bpm > 0
                              ? `${serverResult.heartRate.bpm} bpm`
                              : '--'}
                          </p>
                        </div>

                        <div>
                          <p className="text-xs text-purple-200">Server HRV</p>
                          <p className="text-xl font-bold text-white">
                            {serverResult.hrv.sdnn > 0
                              ? `${serverResult.hrv.sdnn} ms`
                              : '--'}
                          </p>
                        </div>
                      </div>

                      {heartRate.bpm > 0 && serverResult.heartRate.bpm > 0 && (
                        <div className="mt-2 border-t border-white/20 pt-2 text-center">
                          <p className="text-xs text-purple-200">
                            Difference:{' '}
                            {Math.abs(
                              heartRate.bpm - serverResult.heartRate.bpm,
                            ).toFixed(1)}{' '}
                            bpm
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
}
