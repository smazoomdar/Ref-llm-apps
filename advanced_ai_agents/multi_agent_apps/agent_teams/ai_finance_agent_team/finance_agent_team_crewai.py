import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, DuckDuckGoSearchRun
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Define Custom Tools for yfinance
class YahooFinanceTool(BaseTool):
    name: str = "Yahoo Finance Tool"
    description: str = "A tool to fetch financial data from Yahoo Finance."

    def _run(self, ticker: str, mode: str):
        stock = yf.Ticker(ticker)
        if mode == "info":
            return stock.info
        elif mode == "news":
            return stock.news
        elif mode == "recommendations":
            return stock.recommendations
        elif mode == "history":
            return stock.history(period="1y").to_dict() # Get 1 year of historical data
        else:
            return "Invalid mode. Available modes: info, news, recommendations, history."

class GetStockPriceTool(YahooFinanceTool):
    name: str = "Get Stock Price"
    description: str = "Fetches the current stock price and other info for a given ticker."

    def _run(self, ticker: str):
        return super()._run(ticker, mode="info")

class GetCompanyInfoTool(YahooFinanceTool):
    name: str = "Get Company Info"
    description: str = "Fetches detailed company information for a given ticker."

    def _run(self, ticker: str):
        return super()._run(ticker, mode="info")

class GetCompanyNewsTool(YahooFinanceTool):
    name: str = "Get Company News"
    description: str = "Fetches recent news articles for a given company ticker."

    def _run(self, ticker: str):
        return super()._run(ticker, mode="news")

class GetAnalystRecommendationsTool(YahooFinanceTool):
    name: str = "Get Analyst Recommendations"
    description: str = "Fetches analyst recommendations for a given company ticker."

    def _run(self, ticker: str):
        return super()._run(ticker, mode="recommendations")


# Define Agents
web_search_agent = Agent(
    role='Web Search Agent',
    goal='Search the web for relevant information on a given company.',
    backstory='An AI agent skilled in using web search tools to find up-to-date information.',
    verbose=True,
    allow_delegation=False,
    tools=[DuckDuckGoSearchRun()]
)

financial_data_agent = Agent(
    role='Financial Data Agent',
    goal='Fetch and analyze financial data for a given company using yfinance.',
    backstory='An AI agent specialized in using yfinance to gather financial information and are adept at presenting financial summaries in clear, well-structured tables.',
    verbose=True,
    allow_delegation=False,
    tools=[
        GetStockPriceTool(),
        GetCompanyInfoTool(),
        GetCompanyNewsTool(),
        GetAnalystRecommendationsTool()
    ]
)

# Define Tasks
search_company_task = Task(
    description='Search for information about {company_ticker} on the web.',
    expected_output='A summary of relevant web search results for {company_ticker}.',
    agent=web_search_agent
)

get_financial_data_task = Task(
    description='Fetch financial data for {company_ticker} using yfinance tools.',
    expected_output='A compilation of stock price, company info, news, and analyst recommendations for {company_ticker}.',
    agent=financial_data_agent
)

# Define Crew
financial_crew = Crew(
    agents=[web_search_agent, financial_data_agent],
    tasks=[search_company_task, get_financial_data_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    company_ticker = input("Enter the company ticker symbol (e.g., AAPL, MSFT): ")
    inputs = {'company_ticker': company_ticker}

    print(f"\nStarting financial analysis for {company_ticker}...\n")

    result = financial_crew.kickoff(inputs=inputs)

    print(f"\nFinancial Analysis for {company_ticker}:\n")
    print(result)
