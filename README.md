# 📈 AI Financial Analyst Assistant

This project is a powerful, multi-agent financial analysis tool built using CrewAI and Streamlit. It takes a company name or stock ticker, orchestrates a team of AI agents to gather and analyze data from various sources, and presents a comprehensive financial report through an easy-to-use web interface.

## ✨ Features

- **Interactive Web UI**: A clean and simple user interface built with Streamlit.
- **Ticker Search**: Automatically finds the correct stock ticker for a given company name using the Alpha Vantage API. It can even handle ambiguity by letting the user choose from multiple matches.
- **Company Profiling**: Gathers essential company information, including its business summary, sector, industry, and website.
- **In-depth Financial Analysis**:
    - Fetches the latest annual financial statements (Income, Balance Sheet, Cash Flow) from Yahoo Finance.
    - Calculates key financial ratios like Gross Profit Margin, Net Profit Margin, Debt-to-Equity, and Current Ratio.
    - Provides a summary of the company's financial health.
- **Stock Performance Review**: Analyzes the stock's historical performance over the last year, highlighting highs, lows, and recent prices.
- **News & Sentiment Analysis**: Scans recent news and press releases for the company using the Serper API and determines the overall sentiment.
- **Comprehensive Reporting**: A final agent synthesizes all the gathered information into a well-structured and easy-to-read report in Markdown format.
- **API Caching**: Implements caching for financial data API calls to improve performance and avoid redundant requests.

## ⚙️ How It Works: The AI Crew

The application leverages [CrewAI] to simulate a team of specialized AI agents, each with a specific role. The process is sequential, with the output of one task often forming the context for the next.

1.  **Company Identifier Agent**: Takes the user's input and uses the `TickerSearchTool` to find the correct stock ticker.
2.  **Lead Company Researcher Agent**: Uses the identified ticker to fetch general company information with the `CompanyInfoTool`.
3.  **Quantitative Financial Analyst Agent**: Analyzes financial statements obtained via the `CompanyFinancialsTool` and calculates key ratios.
4.  **Stock Performance Analyst Agent**: Fetches and summarizes the 1-year stock history using the `HistoricalStockDataTool`.
5.  **Market News Interpreter Agent**: Gathers and analyzes recent news sentiment using the `SerperDevTool`.
6.  **Chief Financial Reporter Agent**: Consolidates all the findings into the final, structured report.

## 🛠️ Tech Stack

- **Frameworks**: CrewAI, Streamlit
- **LLM**: Google Gemini (via LiteLLM)
- **Data Sources & Tools**:
    - yfinance: For company info, financials, and historical stock data.
    - Alpha Vantage API: For searching ticker symbols.
    - Serper API: For real-time news searches.
- **Core Libraries**: `pydantic`, `requests`, `cachetools`, `python-dotenv`

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8 or higher.
- Git for cloning the repository.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/AI_FinancialAnalyst.git
cd AI_FinancialAnalyst-main
```

### 3. Set Up a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Create a `requirements.txt` file with the following content and then run the installation command.

**`requirements.txt`:**
```
crewai
crewai-tools
streamlit
python-dotenv
yfinance
requests
cachetools
pydantic
google-generativeai
litellm
pysqlite3-binary
```

**Installation Command:**
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys

The application requires API keys from three services. Create a file named `.env` in the root of the project directory (`AI_FinancialAnalyst-main/`) and add your keys.

```env
# .env

# 1. Google Gemini API Key (for the LLM)
# Get it from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# 2. Alpha Vantage API Key (for Ticker Search)
# Get a free key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY="YOUR_ALPHA_VANTAGE_API_KEY"

# 3. Serper API Key (for News Search)
# Get a free key from: https://serper.dev/
SERPER_API_KEY="YOUR_SERPER_API_KEY"
```

### 6. Run the Application

Once the dependencies are installed and your `.env` file is configured, start the Streamlit app:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application's UI.

## Usage

1.  Enter a company name (e.g., "Microsoft") or a ticker symbol (e.g., "MSFT") in the sidebar.
2.  Click the "Analyze Company" button.
3.  If you entered a name, you may be prompted to select the correct company from a list of search results.
4.  Wait for the AI agents to complete their analysis. The progress will be shown in your terminal.
5.  View the final, comprehensive report directly in the web app.

