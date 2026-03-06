// app/types.ts — shared types for PPG and heart rate

export interface Valley {
  index: number;
  value: number;
}

export interface HeartRateResult {
  bpm: number;
  confidence: number;
}

export interface HRVResult {
  sdnn: number;
  confidence: number;
}
