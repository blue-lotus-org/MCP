"""This is an example file, not a real world applicaion"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import asyncio
import json

app = FastAPI(title="Minimal MCP API", version="1.0")

# --- Data Models ---
class MCPRequest(BaseModel):
    model: str
    text: str

# --- Simulated Async Model Responses ---
async def mistral_mock(text):
    await asyncio.sleep(0.5)  # Simulate network delay
    return f"Mistral AI Response: '{text}'"

async def gemini_mock(text):
    await asyncio.sleep(0.5)  # Simulate network delay
    return f"Gemini AI Response: '{text}'"

# --- Unified Model Processing ---
async def process_mcp(model, text):
    if model == "mistral":
        response = await mistral_mock(text)
    elif model == "gemini":
        response = await gemini_mock(text)
    elif model == "both":
        mistral_res, gemini_res = await asyncio.gather(mistral_mock(text), gemini_mock(text))
        response = {"mistral": mistral_res, "gemini": gemini_res}
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    return {"request_id": str(uuid.uuid4()), "model": model, "response": response}

# --- FastAPI Endpoints ---
@app.post("/mcp")
async def mcp_handler(request: MCPRequest):
    return await process_mcp(request.model, request.text)

# --- Run Server (Only for Local Testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the API Locally
"""
uvicorn mcp_api:app --host 0.0.0.0 --port 8000 --reload
"""

# Send a Request (cURL or Postman)
"""
curl -X POST "http://127.0.0.1:8000/mcp" \
     -H "Content-Type: application/json" \
     -d '{"model": "both", "text": "Explain the impact of AI on business"}'
"""

# Example API Response
"""
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "model": "both",
  "response": {
    "mistral": "Mistral AI Response: 'Explain the impact of AI on business'",
    "gemini": "Gemini AI Response: 'Explain the impact of AI on business'"
  }
}
"""
