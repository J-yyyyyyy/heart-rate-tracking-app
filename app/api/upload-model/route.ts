import { NextResponse } from 'next/server';

const flaskUrl = process.env.NEXT_PUBLIC_FLASK_URL || 'http://127.0.0.1:5000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const res = await fetch(`${flaskUrl}/upload-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ success: false, error: 'Request failed' }, { status: 502 });
  }
}
