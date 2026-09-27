import os
import json
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NexaFi AI Assistant API",
    description="AI-powered assistant for NexaFi DeFi platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    logger.warning("ANTHROPIC_API_KEY not found in environment variables!")
else:
    logger.info("Anthropic API key loaded successfully.")

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class DashboardData(BaseModel):
    tvl: int
    apy: float
    apy_change: float
    users: int
    portfolio_growth: float
    chart_data: List[Dict[str, Any]]
    ai_insight: str


def generate_chart_data():
    base_apy = 14.5
    data = []
    for i in range(30):
        value = base_apy + random.uniform(-0.8, 0.8)
        data.append({"day": f"{i}d", "value": round(value, 2)})
    return data


def get_dashboard_data():
    tvl = random.randint(120000000, 200000000)
    apy = round(random.uniform(13.5, 15.5), 2)
    apy_change = round(random.uniform(-2.0, 2.5), 2)
    users = random.randint(45000, 55000)
    portfolio_growth = round(random.uniform(1.5, 4.2), 1)
    chart_data = generate_chart_data()

    insight = (
        f"ETH is up {round(random.uniform(0.5, 4.0), 2)}% in 24h. "
        f"DeFi TVL sits at ${tvl:,}. "
        f"Current blended APY is {apy}% — "
        f"rebalancing {random.randint(10, 30)}% into stNEXA could lift your yield."
    )

    return DashboardData(
        tvl=tvl,
        apy=apy,
        apy_change=apy_change,
        users=users,
        portfolio_growth=portfolio_growth,
        chart_data=chart_data,
        ai_insight=insight,
    )


NEXAFI_SYSTEM_PROMPT = """You are Nexa AI, the friendly and knowledgeable assistant for NexaFi — a decentralized finance platform for trading, staking, and growing crypto with AI-powered insights.

Core facts about NexaFi:
- Non-custodial: users keep control of their private keys
- Cross-chain: supports Ethereum, Arbitrum, Base, and Solana
- AI-powered trading signals and auto-strategies
- Secure staking pools with up to 14% APY
- Web3 wallet connections: MetaMask, WalletConnect, Rainbow (no sign-up required)
- Institutional API with REST endpoints and webhooks

Communication style:
- Be concise, helpful, and professional
- Explain DeFi concepts in simple terms
- Never give financial advice; always include a disclaimer when discussing yields or investments
- If asked about security, emphasize non-custodial nature and user key ownership
- If asked about wallet connections, mention supported wallets
- If asked about chains, mention Ethereum, Arbitrum, Base, Solana
- If asked about pricing, mention: Starter (Free), Pro ($29/month), Enterprise (Custom)

Current dashboard context: {dashboard_context}

When responding:
1. Keep answers under 150 words unless the user asks for detail
2. Use bullet points for lists
3. Be encouraging but cautious about crypto risks
4. If you don't know something, say so honestly
5. Always end investment-related answers with a risk disclaimer

Do not make up specific token prices, contract addresses, or transaction data."""


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "NexaFi AI Assistant API",
        "version": "1.0.0",
        "endpoints": ["/api/chat", "/api/dashboard"],
    }


@app.get("/api/dashboard")
async def dashboard():
    try:
        data = get_dashboard_data()
        return data.model_dump()
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard data")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not anthropic_client:
        logger.error("Anthropic client not initialized")
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please check API configuration."
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        dashboard = get_dashboard_data()
        dashboard_context = json.dumps({
            "tvl": dashboard.tvl,
            "apy": dashboard.apy,
            "apy_change": dashboard.apy_change,
            "users": dashboard.users,
            "portfolio_growth": dashboard.portfolio_growth,
            "ai_insight": dashboard.ai_insight,
        })

        system_prompt = NEXAFI_SYSTEM_PROMPT.format(dashboard_context=dashboard_context)

        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        logger.info(f"Processing chat request with {len(anthropic_messages)} messages")

        response = anthropic_client.messages.create(
            model="claude-4-5-haiku-latest",
            max_tokens=400,
            system=system_prompt,
            messages=anthropic_messages,
        )

        reply = response.content[0].text

        logger.info("Chat response generated successfully")

        return {
            "reply": reply,
            "model": "claude-4-5-haiku-latest",
            "tokens_used": response.usage.output_tokens + response.usage.input_tokens,
        }

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
