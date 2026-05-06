from dotenv import load_dotenv
import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from openai import OpenAI
import requests

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
#fav_stocks = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]

# ── Tool definition (what the LLM sees) ──────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_stock_news",
            "description": "Search the web for the latest news about a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol e.g. AAPL"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# ── Tool implementation (actual logic) ───────────────────────────────
def search_stock_news(ticker: str) -> str:
    """Real news fetch using NewsAPI"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "pageSize": 3,
        "apiKey": os.environ["NEWS_API_KEY"]
    }
    res = requests.get(url, params=params).json()
    articles = res.get("articles", [])
    if not articles:
        return "No news found."
    
    # Return top 3 headlines to the LLM
    headlines = []
    for a in articles[:3]:
        headlines.append(f"- {a['title']} ({a['url']})")
    return "\n".join(headlines)

# ── Tool dispatcher ───────────────────────────────────────────────────
def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_stock_news":
        return search_stock_news(tool_args["ticker"])
    return "Unknown tool"

# ── Agentic loop ─────────────────────────────────────────────────────
def run_agent(stock: str) -> dict:
    print(f"\nProcessing {stock}...")
    messages = [
        {
            "role": "system",
            "content": "You are a stock news analyst. Use the search tool to find real news, then summarize and analyze sentiment."
        },
        {
            "role": "user",
            "content": f"Find the latest news for {stock} stock. Summarize it in one sentence and tell me if the sentiment is positive, negative, or neutral."
        }
    ]

    # Step 1: LLM decides to call a tool
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=512,
    )

    message = response.choices[0].message

    # Step 2: If LLM called a tool, run it and feed result back
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"  → Agent called tool: {tool_name}({tool_args})")
        tool_result = run_tool(tool_name, tool_args)
        print(f"  → Tool returned: {tool_result[:80]}...")

        # Append tool call + result to message history
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # Step 3: LLM reasons over real data and gives final answer
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
        )
        final_output = final_response.choices[0].message.content.strip()
    else:
        # LLM answered directly without tool (fallback)
        final_output = message.content.strip()

    # Parse sentiment
    lower = final_output.lower()
    if "positive" in lower:
        emoji = "😊"
        sentiment = "Positive"
    elif "negative" in lower:
        emoji = "😞"
        sentiment = "Negative"
    else:
        emoji = "😐"
        sentiment = "Neutral"

    return {
        "Stock": stock,
        "News Summary": final_output,
        "Sentiment": f"{sentiment} {emoji}",
    }

async def main():
    user_input = input("Enter stock tickers separated by commas (e.g. AAPL, GOOGL, TSLA): ")
    fav_stocks = [stock.strip().upper() for stock in user_input.split(",")]
    
    print(f"\nFetching news for: {fav_stocks}")
    results = [run_agent(stock) for stock in fav_stocks]

    df = pd.DataFrame(results)
    output_path = Path("stock_news_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())