import { NextResponse } from 'next/server';

const flaskUrl = process.env.FLASK_URL || 'http://127.0.0.1:5000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (!body.ppgData || !Array.isArray(body.ppgData)) {
      return NextResponse.json(
        { error: 'Missing ppgData array' },
        { status: 400 },
      );
    }
    const res = await fetch(`${flaskUrl}/infer-quality`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ppgData: body.ppgData }),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: 'Request failed' },
      { status: 502 },
    );
  }
}
