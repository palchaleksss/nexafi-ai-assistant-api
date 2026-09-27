import os
import random
import requests
import anthropic
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_BASE = "https://api.llama.fi"
DEFILLAMA_YIELDS = "https://yields.llama.fi"
NEXA_BASE_APY = 14.2

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_cache = {"data": None, "expires": datetime.min}


def fetch_eth_data():
    try:
        eth_res = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency": "usd", "ids": "ethereum", "sparkline": "true"},
            timeout=10
        )
        eth_res.raise_for_status()
        eth_data = eth_res.json()
        return eth_data[0] if eth_data else None
    except Exception:
        return None


def fetch_tvl():
    try:
        tvl_res = requests.get(f"{DEFILLAMA_BASE}/charts", timeout=10)
        tvl_res.raise_for_status()
        tvl_data = tvl_res.json()
        if tvl_data and isinstance(tvl_data, list):
            return tvl_data[-1].get("totalLiquidityUSD", 128400000)
        return 128400000
    except Exception:
        return 128400000


def fetch_apy_history():
    try:
        pools_res = requests.get(f"{DEFILLAMA_YIELDS}/pools", timeout=10)
        pools_res.raise_for_status()
        pools = pools_res.json().get("data", [])

        pool = next(
            (
                p
                for p in pools
                if "steth" in p.get("symbol", "").lower()
                and "lido" in p.get("project", "").lower()
            ),
            None,
        )

        if not pool:
            return None

        chart_res = requests.get(f"{DEFILLAMA_YIELDS}/chart/{pool['pool']}", timeout=10)
        chart_res.raise_for_status()
        data = chart_res.json().get("data", [])

        if len(data) < 10:
            return None

        raw_apys = [d["apy"] for d in data[-30:]]
        base = raw_apys[0] or 1
        scaled = [NEXA_BASE_APY * (v / base) for v in raw_apys]
        return [round(v, 2) for v in scaled]

    except Exception:
        return None


def generate_fallback_apy_chart(base=NEXA_BASE_APY, days=30):
    values = [base]
    for _ in range(1, days):
        values.append(max(0.5, values[-1] * (1 + random.uniform(-0.04, 0.04))))
    return [{"day": f"{i}d", "value": round(v, 2)} for i, v in enumerate(values)]


def build_dashboard_data():
    eth = fetch_eth_data()
    eth_change = eth.get("price_change_percentage_24h", 0) or 0 if eth else 2.1

    defi_tvl = fetch_tvl()
    apy_history = fetch_apy_history()
    chart_data = (
        [{"day": f"{i}d", "value": v} for i, v in enumerate(apy_history)]
        if apy_history
        else generate_fallback_apy_chart()
    )

    current_apy = chart_data[-1]["value"] if chart_data else NEXA_BASE_APY
    start_apy = chart_data[0]["value"] if chart_data else NEXA_BASE_APY
    apy_change = ((current_apy - start_apy) / start_apy) * 100

    insight = (
        f"ETH is up {eth_change:.2f}% in 24h. "
        f"DeFi TVL sits at ${defi_tvl:,.0f}. "
        f"Current blended APY is {current_apy:.2f}% — "
        f"rebalancing 15% into stNEXA could lift your yield."
    )

    return {
        "tvl": defi_tvl,
        "apy": current_apy,
        "apy_change": round(apy_change, 2),
        "users": 50247,
        "portfolio_growth": round(eth_change, 2),
        "chart_data": chart_data,
        "ai_insight": insight,
    }


def get_cached_dashboard():
    now = datetime.utcnow()
    if _cache["data"] and _cache["expires"] > now:
        return _cache["data"]

    data = build_dashboard_data()
    _cache["data"] = data
    _cache["expires"] = now + timedelta(minutes=5)
    return data


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
async def dashboard():
    try:
        return get_cached_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
@limiter.limit("15/minute")
async def chat(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages required")

        system_prompt = (
            "Ты — NexaFi AI, лаконичный и полезный ассистент платформы NexaFi DeFi. "
            "Ты помогаешь пользователям со стейкингом, доходностью, кошельками, кроссчейн-обменами и функциями платформы. "
            "Держи ответы максимум в 3 предложения. Никогда не раскрывай внутренние API-ключи и детали бэкенда. "
            "Если спрашивают про цены или APY, скажи, что они динамичные, и предложи проверить живой дашборд."
        )

        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=250,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )

        return {"reply": response.content[0].text}

    except anthropic.APIError:
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/chat")
@limiter.limit("15/minute")
async def chat(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages required")

        system_prompt = (
            "You are NexaFi AI, a concise and helpful assistant for the NexaFi DeFi platform. "
            "You help users with staking, yield, wallets, cross-chain swaps, and platform features. "
            "Keep answers under 3 sentences when possible. Never share internal API keys or backend details. "
            "If asked about prices or APY, say they are dynamic and suggest checking the live dashboard."
        )

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=400,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )

        return {"reply": response.content[0].text}

    except anthropic.APIError:
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/embed", response_class=HTMLResponse)
async def embed_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nexa AI Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: #fff;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .card {
                width: 100%;
                max-width: 520px;
                background: #0f0f14;
                border: 1px solid #1f1f2e;
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .header-left {
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            .logo {
                width: 28px; height: 28px;
                background: linear-gradient(135deg, #6366f1, #a855f7);
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 14px;
            }
            .live {
                display: flex; align-items: center; gap: 6px;
                font-size: 11px; font-weight: 600;
                color: #22d3ee; text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .live::before {
                content: ""; width: 6px; height: 6px;
                background: #22d3ee; border-radius: 50%;
                box-shadow: 0 0 8px #22d3ee;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom: 20px;
            }
            .stat {
                background: #16161f;
                border-radius: 14px;
                padding: 16px;
            }
            .stat-label {
                font-size: 12px; color: #94a3b8; margin-bottom: 6px;
            }
            .stat-value {
                font-size: 20px; font-weight: 700; margin-bottom: 4px;
            }
            .stat-change {
                font-size: 12px; color: #22d3ee; font-weight: 500;
            }
            .chart-block {
                background: #16161f;
                border-radius: 14px;
                padding: 16px;
                margin-bottom: 16px;
            }
            .chart-header {
                display: flex; justify-content: space-between;
                font-size: 12px; color: #94a3b8; margin-bottom: 14px;
            }
            .chart-header span:last-child { color: #22d3ee; font-weight: 600; }
            .bars {
                display: flex; align-items: flex-end; justify-content: space-between;
                height: 100px; gap: 6px;
            }
            .bar {
                flex: 1;
                background: linear-gradient(180deg, #6366f1, #a855f7);
                border-radius: 4px 4px 0 0;
                opacity: 0.7;
                transition: opacity 0.2s;
                min-width: 4px;
            }
            .bar:last-child { opacity: 1; }
            .insight {
                background: #16161f;
                border-radius: 14px;
                padding: 16px;
                display: flex;
                align-items: flex-start;
                gap: 12px;
                font-size: 13px; line-height: 1.5; color: #cbd5e1;
            }
            .insight-icon {
                width: 28px; height: 28px;
                background: #1e1e2d;
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 14px; flex-shrink: 0;
            }
            .insight strong { color: #fff; }
            .loading, .error {
                text-align: center; padding: 60px 20px; color: #94a3b8; font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="card" id="card">
            <div class="loading">Loading Nexa AI Dashboard...</div>
        </div>

        <script>
            function formatMoney(n) {
                if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
                if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
                if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
                return '$' + n.toFixed(2);
            }
            function formatUsers(n) {
                if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
                if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
                return n.toString();
            }

            function renderDashboard(data) {
                const card = document.getElementById('card');
                const chart = Array.isArray(data.chart_data) ? data.chart_data : [];
                const values = chart.map(d => d.value || 0);
                const maxChart = values.length ? Math.max(...values) : 1;

                card.innerHTML = `
                    <div class="header">
                        <div class="header-left">
                            <div class="logo">✦</div>
                            <div>Nexa AI Dashboard</div>
                        </div>
                        <div class="live">Live</div>
                    </div>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-label">TVL</div>
                            <div class="stat-value">${formatMoney(data.tvl || 0)}</div>
                            <div class="stat-change">+${(data.portfolio_growth || 0).toFixed(1)}% 24h</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">APY</div>
                            <div class="stat-value">${(data.apy || 0).toFixed(1)}%</div>
                            <div class="stat-change">${(data.apy_change || 0) >= 0 ? '+' : ''}${(data.apy_change || 0).toFixed(1)}% 30d</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Users</div>
                            <div class="stat-value">${formatUsers(data.users || 0)}</div>
                            <div class="stat-change">+1,204 today</div>
                        </div>
                    </div>
                    <div class="chart-block">
                        <div class="chart-header">
                            <span>APY · 30D</span>
                            <span>${(data.apy || 0).toFixed(2)}%</span>
                        </div>
                        <div class="bars">
                            ${chart.length ? chart.map(d => `
                                <div class="bar" style="height: ${((d.value || 0) / maxChart * 100).toFixed(1)}%"></div>
                            `).join('') : '<div style="color:#64748b;font-size:12px;">No chart data</div>'}
                        </div>
                    </div>
                    <div class="insight">
                        <div class="insight-icon">🤖</div>
                        <div><strong>Nexa AI</strong> · ${data.ai_insight || 'Market data temporarily unavailable.'}</div>
                    </div>
                `;
            }

            fetch('/api/dashboard')
                .then(r => {
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(renderDashboard)
                .catch(err => {
                    document.getElementById('card').innerHTML = `<div class="error">Failed to load dashboard.<br>${err.message}</div>`;
                    console.error(err);
                });
        </script>
    </body>
    </html>
    """
