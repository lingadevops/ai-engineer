from dotenv import load_dotenv
import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from openai import OpenAI
import requests
import yfinance as yf

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

price_tool = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker, day change, high and low price in day.",
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

def get_stock_price(ticker: str) -> str:
    """Fetch current stock price using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return "No price data found."
        current_price = data['Close'][0]
        day_change = data['Close'][0] - data['Open'][0]
        high = data['High'][0]
        low = data['Low'][0]
        return f"Current price: ${current_price:.2f}, Day change: ${day_change:.2f}, High: ${high:.2f}, Low: ${low:.2f}"
    except Exception as e:
        return f"Error fetching price: {str(e)}"


# ── Tool dispatcher ───────────────────────────────────────────────────
def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_stock_news":
        return search_stock_news(tool_args["ticker"])
    elif tool_name == "get_stock_price":
        return get_stock_price(tool_args["ticker"])
    return "Unknown tool"

# ── Agentic loop ─────────────────────────────────────────────────────
def run_agent(stock: str) -> dict:
    print(f"\nProcessing {stock}...")
    messages = [
        {
            "role": "system",
            "content": "system_prompt"
        },
        {
            "role": "user",
            "content": "system_prompt"
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

        print(f" [tool called] {tool_name}({tool_args})")
        tool_result = run_tool(tool_name, tool_args)
        #print(f"  → Tool returned: {tool_result[:80]}...")

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
def news_agent(ticker: str) -> str:
    print(f"  [News Agent] fetching news for {ticker}...")
    return run_agent(
        system_prompt="You are a financial news analyst. Use the tool to fetch real news, then summarize the top headline in 1-2 sentences.",
        user_prompt=f"Get the latest news for {ticker} stock.",
        tools=tools
    )


def price_agent(ticker: str) -> str:
    print(f"  [Price Agent] fetching price for {ticker}...")
    return run_agent(
        system_prompt="You are a stock price analyst. Use the tool to get the current price and summarize the price movement briefly.",
        user_prompt=f"Get the current stock price and day performance for {ticker}.",
        tools=price_tool
    )


def sentiment_agent(ticker: str, news_summary: str) -> str:
    print(f"  [Sentiment Agent] analyzing sentiment for {ticker}...")
    # This agent needs no tool — it reasons over the news summary directly
    return run_agent(
        system_prompt="You are a sentiment analyst. Given a news summary, respond with ONLY one word: positive, negative, or neutral.",
        user_prompt=f"What is the sentiment of this news about {ticker}? '{news_summary}'",
        tools=[]
    )


def report_agent(ticker: str, news: str, price: str, sentiment: str) -> str:
    print(f"  [Report Agent] generating final report for {ticker}...")
    # Combines all agent outputs into a final recommendation
    return run_agent(
        system_prompt="You are a senior investment analyst. Given news, price and sentiment, write a concise 3-4 sentence investment summary with a buy/hold/sell suggestion.",
        user_prompt=f"""
Ticker: {ticker}
News: {news}
Price: {price}
Sentiment: {sentiment}

Write a brief investment summary.
        """,
        tools=[]  # no tools needed — just reasoning
    )

def orchestrator(ticker: str) -> dict:
    """
    Runs all agents in order for a single stock.
    Each agent's output feeds into the next.
    """
    print(f"\n{'='*50}")
    print(f"  Orchestrating agents for {ticker}")
    print(f"{'='*50}")

    news         = news_agent(ticker)
    price        = price_agent(ticker)
    sentiment    = sentiment_agent(ticker, news)
    report       = report_agent(ticker, news, price, sentiment)

    # Sentiment emoji
    s = sentiment.lower()
    if "positive" in s:
        emoji = "😊"
    elif "negative" in s:
        emoji = "😞"
    else:
        emoji = "😐"

    return {
        "Stock":        ticker,
        "News":         news,
        "Price":        price,
        "Fundamentals": fundamentals,
        "Sentiment":    f"{sentiment.capitalize()} {emoji}",
        "Report":       report,
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