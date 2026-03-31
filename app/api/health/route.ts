import { NextResponse } from 'next/server';

const flaskUrl = process.env.NEXT_PUBLIC_FLASK_URL || 'http://127.0.0.1:5000';

export async function GET() {
  try {
    const res = await fetch(`${flaskUrl}/health`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ ok: false }, { status: 502 });
  }
}
