"""This is just an example, not real world application"""
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import httpx
import asyncio
import os
import uuid

# --- FastAPI App ---
app = FastAPI(title="Enhanced MCP API", version="2.0")

# --- Configurations ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your-mistral-key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key")
ALLOWED_API_KEYS = {"secret-key-123", "another-key"}  # API Key Auth

# --- Database Setup (SQLite) ---
DATABASE_URL = "sqlite:///./mcp.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Model ---
class MCPLog(Base):
    __tablename__ = "mcp_requests"
    request_id = Column(String, primary_key=True, index=True)
    model = Column(String, index=True)
    input_text = Column(Text)
    response = Column(Text)

Base.metadata.create_all(bind=engine)

# --- API Request Model ---
class MCPRequest(BaseModel):
    model: str
    text: str

# --- Dependency: Database Session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Dependency: API Key Auth ---
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Asynchronous Calls to AI APIs ---
async def call_mistral(text):
    url = "https://api.mistral.ai/generate"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    payload = {"prompt": text, "max_tokens": 100}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        return response.json().get("text", "Mistral API Error")

async def call_gemini(text):
    url = "https://api.gemini.com/v1/generate"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"input": text}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        return response.json().get("output", "Gemini API Error")

# --- Unified Processing ---
async def process_mcp(model, text):
    if model == "mistral":
        response = await call_mistral(text)
    elif model == "gemini":
        response = await call_gemini(text)
    elif model == "both":
        mistral_res, gemini_res = await asyncio.gather(call_mistral(text), call_gemini(text))
        response = {"mistral": mistral_res, "gemini": gemini_res}
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    return {"request_id": str(uuid.uuid4()), "model": model, "response": response}

# --- API Endpoints ---
@app.post("/mcp", dependencies=[Depends(verify_api_key)])
async def mcp_handler(request: MCPRequest, db=Depends(get_db)):
    result = await process_mcp(request.model, request.text)

    # Log to Database
    db_entry = MCPLog(
        request_id=result["request_id"],
        model=result["model"],
        input_text=request.text,
        response=str(result["response"])
    )
    db.add(db_entry)
    db.commit()

    return result

# Run API Server
"""
uvicorn mcp_api:app --host 0.0.0.0 --port 8000 --reload
"""

# Send Request with API Key
"""
curl -X POST "http://127.0.0.1:8000/mcp" \
     -H "Content-Type: application/json" \
     -H "x-api-key: secret-key-123" \
     -d '{"model": "both", "text": "Explain the impact of AI"}'
"""

# Example API Response
"""
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "model": "both",
  "response": {
    "mistral": "Mistral API Response: 'Explain the impact of AI'",
    "gemini": "Gemini API Response: 'Explain the impact of AI'"
  }
}
"""
