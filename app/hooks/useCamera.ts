'use client';

import { useState, useRef, useEffect } from 'react';

export default function useCamera() {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Camera access failed');
    }
  }

  function stopCamera() {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
    }
  }

  useEffect(() => {
    if (isRecording) startCamera();
    else stopCamera();
    return () => stopCamera();
  }, [isRecording]);

  return { videoRef, canvasRef, isRecording, setIsRecording, error };
}
