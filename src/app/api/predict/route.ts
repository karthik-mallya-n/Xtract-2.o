import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const body = await request.json();
    
    // Forward the request to Flask backend
    const flaskUrl = process.env.NEXT_PRIVATE_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://backend:5000';
    const response = await fetch(`${flaskUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Flask prediction request failed:', errorText);
      return NextResponse.json(
        { 
          success: false, 
          error: `Flask server error: ${response.status}` 
        }, 
        { status: response.status }
      );
    }
    
    // Forward the Flask response back to the frontend
    const result = await response.json();
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('Prediction proxy error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error occurred' 
      }, 
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json(
    { 
      success: false, 
      error: 'Only POST requests are allowed for predictions' 
    }, 
    { status: 405 }
  );
}