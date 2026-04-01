import { NextResponse } from 'next/server';

const flaskUrl = process.env.NEXT_PUBLIC_FLASK_URL || 'http://127.0.0.1:5000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (!body.ppgData || !Array.isArray(body.ppgData) || !body.label) {
      return NextResponse.json(
        { success: false, error: 'Missing ppgData or label' },
        { status: 400 },
      );
    }
    if (body.label !== 'good' && body.label !== 'bad') {
      return NextResponse.json(
        { success: false, error: 'Label must be good or bad' },
        { status: 400 },
      );
    }
    const res = await fetch(`${flaskUrl}/save-labeled-segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { success: false, error: 'Request failed' },
      { status: 502 },
    );
  }
}
