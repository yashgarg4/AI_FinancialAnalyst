import os
import yfinance as yf
import requests # For making API calls
from crewai.tools import BaseTool
from typing import Type, Any
from cachetools import TTLCache
from pydantic import BaseModel, Field # Using standard Pydantic import

class TickerInput(BaseModel):
    """Input for financial tools, requiring a company ticker."""
    ticker: str = Field(..., description="The stock ticker symbol of the company (e.g., AAPL for Apple).")

class CompanyNameInput(BaseModel):
    """Input for ticker search tool, requiring a company name or keyword."""
    company_name: str = Field(..., description="The name or keyword of the company to search for (e.g., Apple Inc).")

class CompanyInfoTool(BaseTool):
    name: str = "Company Information Tool"
    description: str = (
        "Fetches general information about a publicly traded company using its stock ticker. "
        "This includes company profile, sector, industry, full-time employees, and key executives."
    )
    args_schema: Type[BaseModel] = TickerInput
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=1800)) # Cache for 30 minutes

    def _run(self, ticker: str) -> dict:
        if ticker in self.cache:
            print(f"Cache hit for CompanyInfoTool: {ticker}")
            return self.cache[ticker]
        try:
            print(f"Cache miss for CompanyInfoTool: {ticker}. Fetching from API.")
            company = yf.Ticker(ticker)
            info = company.info
            # Select relevant fields to avoid overwhelming the LLM
            relevant_info = {
                "company_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "website": info.get("website"),
                "long_business_summary": info.get("longBusinessSummary"),
                "full_time_employees": info.get("fullTimeEmployees"),
                # "company_officers": info.get("companyOfficers") # Often too verbose
            }
            self.cache[ticker] = relevant_info
            return relevant_info
        except Exception as e:
            return {"error": f"Failed to fetch company info for {ticker}: {str(e)}"}

class CompanyFinancialsTool(BaseTool):
    name: str = "Company Financial Statements Tool"
    description: str = (
        "Fetches key financial statements (income statement, balance sheet, cash flow statement - latest annual) "
        "for a publicly traded company using its stock ticker."
    )
    args_schema: Type[BaseModel] = TickerInput
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=1800)) # Cache for 30 minutes

    def _run(self, ticker: str) -> dict:
        if ticker in self.cache:
            print(f"Cache hit for CompanyFinancialsTool: {ticker}")
            return self.cache[ticker]

        def get_latest_annual_data(statement_df):
            if statement_df.empty:
                return "Not available"
            # yfinance columns are timestamps; sort by year and get the latest
            # Assuming columns are sorted chronologically by yfinance, take the last column for latest annual
            latest_column = statement_df.columns[0] # yfinance typically has most recent first
            return statement_df[latest_column].to_dict()

        try:
            print(f"Cache miss for CompanyFinancialsTool: {ticker}. Fetching from API.")
            company = yf.Ticker(ticker)

            income_stmt_annual = get_latest_annual_data(company.income_stmt)
            balance_sheet_annual = get_latest_annual_data(company.balance_sheet)
            cash_flow_annual = get_latest_annual_data(company.cashflow)

            # Extract key figures for easier LLM processing and ratio calculation
            # Ensure to handle cases where keys might be missing (use .get())
            financials = {
                "latest_annual_income_statement_summary": {
                    "Total Revenue": income_stmt_annual.get("Total Revenue") if isinstance(income_stmt_annual, dict) else None,
                    "Gross Profit": income_stmt_annual.get("Gross Profit") if isinstance(income_stmt_annual, dict) else None,
                    "Operating Income": income_stmt_annual.get("Operating Income") if isinstance(income_stmt_annual, dict) else None,
                    "Net Income": income_stmt_annual.get("Net Income") if isinstance(income_stmt_annual, dict) else None,
                } if isinstance(income_stmt_annual, dict) else "Not available",
                "latest_annual_balance_sheet_summary": {
                    "Total Assets": balance_sheet_annual.get("Total Assets") if isinstance(balance_sheet_annual, dict) else None,
                    "Total Liabilities Net Minority Interest": balance_sheet_annual.get("Total Liabilities Net Minority Interest") if isinstance(balance_sheet_annual, dict) else None, # yfinance specific name
                    "Total Equity Gross Minority Interest": balance_sheet_annual.get("Total Equity Gross Minority Interest") if isinstance(balance_sheet_annual, dict) else None, # yfinance specific name
                    "Current Assets": balance_sheet_annual.get("Current Assets") if isinstance(balance_sheet_annual, dict) else None,
                    "Current Liabilities": balance_sheet_annual.get("Current Liabilities") if isinstance(balance_sheet_annual, dict) else None,
                } if isinstance(balance_sheet_annual, dict) else "Not available",
                "latest_annual_cash_flow_summary": {
                    "Operating Cash Flow": cash_flow_annual.get("Operating Cash Flow") if isinstance(cash_flow_annual, dict) else None,
                } if isinstance(cash_flow_annual, dict) else "Not available",
            }
            self.cache[ticker] = financials
            return financials
        except Exception as e:
            return {"error": f"Failed to fetch financial statements for {ticker}: {str(e)}"}

class TickerSearchTool(BaseTool):
    name: str = "Ticker Search Tool"
    description: str = (
        "Searches for stock ticker symbols using a company name or keyword. "
        "Returns a list of potential matches with their ticker symbol, name, region, and match score."
    )
    args_schema: Type[BaseModel] = CompanyNameInput
    alpha_vantage_api_key: str = Field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY"))
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=3600)) # Cache for 1 hour

    def _run(self, company_name: str) -> dict:
        if company_name in self.cache:
            print(f"Cache hit for TickerSearchTool: {company_name}")
            return self.cache[company_name]

        if not self.alpha_vantage_api_key:
            return {"error": "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY environment variable."}
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={self.alpha_vantage_api_key}"
        print(f"Cache miss for TickerSearchTool: {company_name}. Fetching from API.")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors (4XX, 5XX)
            data = response.json()

            if "Error Message" in data:
                return {"error": f"Alpha Vantage API error: {data['Error Message']}"}
            if "Note" in data: # Handles API call frequency limit notes
                return {"api_limit_note": data["Note"], "matches": []}
            
            best_matches = data.get("bestMatches", [])
            if not best_matches: # If bestMatches is an empty list or not found
                return {"message": "No matches found for the given keyword.", "matches": []}
            
            result = {"matches": best_matches}
            self.cache[company_name] = result
            return result
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to search for ticker for {company_name}: {str(e)}"}

class HistoricalStockDataTool(BaseTool):
    name: str = "Historical Stock Data Tool"
    description: str = (
        "Fetches historical stock data (closing prices) for a given ticker over a specified period (e.g., '1y', '6mo', '3mo')."
        "Returns a summary including start date, end date, highest price, lowest price, and current price within that period."
    )
    args_schema: Type[BaseModel] = TickerInput # Re-uses TickerInput, period is handled by LLM in task
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=1800)) # Cache for 30 minutes

    def _run(self, ticker: str, period: str = "1y") -> dict: # Period can be passed if tool is called directly
        cache_key = f"{ticker}_{period}"
        if cache_key in self.cache:
            print(f"Cache hit for HistoricalStockDataTool: {cache_key}")
            return self.cache[cache_key]
        try:
            print(f"Cache miss for HistoricalStockDataTool: {cache_key}. Fetching from API.")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return {"error": f"No historical data found for {ticker} for period {period}."}
            
            summary = {
                "ticker": ticker,
                "period": period,
                "start_date": hist.index[0].strftime('%Y-%m-%d'),
                "end_date": hist.index[-1].strftime('%Y-%m-%d'),
                "highest_price": round(hist['High'].max(), 2),
                "lowest_price": round(hist['Low'].min(), 2),
                "current_price_at_period_end": round(hist['Close'][-1], 2)
            }
            self.cache[cache_key] = summary
            return summary
        except Exception as e:
            return {"error": f"Failed to fetch historical stock data for {ticker}: {str(e)}"}

# Example Usage (for testing the tools independently)
if __name__ == '__main__':
    # Test CompanyInfoTool
    info_tool = CompanyInfoTool()
    print("--- Testing CompanyInfoTool ---")
    aapl_info = info_tool.run(ticker="AAPL")
    print("Info for 'AAPL':", aapl_info)
    msft_info = info_tool.run(ticker="MSFT") 
    print("Info for 'MSFT':", msft_info) 

    # Test CompanyFinancialsTool
    financials_tool = CompanyFinancialsTool()
    print("\n--- Testing CompanyFinancialsTool ---")
    aapl_financials = financials_tool.run(ticker="AAPL")
    print("Financials for 'AAPL':", aapl_financials)

    # Test TickerSearchTool
    # Ensure ALPHA_VANTAGE_API_KEY is set in your .env file for this test to work
    print("\n--- Testing TickerSearchTool ---")
    ticker_search = TickerSearchTool()
    search_result_apple = ticker_search.run(company_name="Apple")
    print("Search for 'Apple':", search_result_apple)
    search_result_tesla = ticker_search.run(company_name="Tesla Inc")
    print("Search for 'Tesla Inc':", search_result_tesla)
    search_result_invalid = ticker_search.run(company_name="NonExistentCompanyXYZ123")
    print("Search for 'NonExistentCompanyXYZ123':", search_result_invalid)

    # Test HistoricalStockDataTool
    print("\n--- Testing HistoricalStockDataTool ---")
    hist_tool = HistoricalStockDataTool()
    aapl_hist_1y = hist_tool.run(ticker="AAPL", period="1y") # Note: period is passed here for direct test
    print("AAPL 1y historical data summary:", aapl_hist_1y)
