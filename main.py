from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nexafi.framer.website"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are NexaFi AI, a helpful assistant for a Web3 finance platform. "
    "Be concise, friendly, and accurate. Explain DeFi, trading, staking, and yields simply. "
    "If the user asks for investment advice, remind them this is not financial advice."
    "You are NexaFi AI... Do not use Markdown formatting like **, *, #, or bullet points. Use plain text only."

)

@app.post("/chat")
async def chat(req: ChatRequest):
    if not CLAUDE_API_KEY:
        raise HTTPException(status_code=500, detail="Claude API key not configured")

    user_messages = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if m.role in ("user", "assistant")
    ]

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": req.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": user_messages,
    }

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(CLAUDE_API_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Claude API error: {resp.status_code} - {resp.text[:200]}"
            )

        data = resp.json()
        answer = data.get("content", [{}])[0].get("text", "No response")

        return {"reply": answer}

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}
