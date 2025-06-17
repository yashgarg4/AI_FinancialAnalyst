try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import litellm
import streamlit as st
from pydantic import BaseModel, Field as PydanticField
from typing import List, Optional
from financial_tools import CompanyInfoTool, CompanyFinancialsTool, TickerSearchTool, HistoricalStockDataTool

# Load environment variables
load_dotenv()

# --- LiteLLM Configuration for Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")

TARGET_GEMINI_MODEL = "gemini-1.5-flash-latest"

# Register the model with LiteLLM
# This allows CrewAI to use "gemini/model_name" as the llm identifier
litellm.register_model({
    "gemini/" + TARGET_GEMINI_MODEL: {
        "model_name": TARGET_GEMINI_MODEL,
        "litellm_provider": "gemini",
        "api_key": GEMINI_API_KEY,
    },
    # Fallback if CrewAI or LiteLLM internally uses a different format
    TARGET_GEMINI_MODEL: {
        "model_name": TARGET_GEMINI_MODEL,
        "litellm_provider": "gemini",
        "api_key": GEMINI_API_KEY,
    }
})

# LLM identifier for CrewAI agents
agent_llm_identifier = f"gemini/{TARGET_GEMINI_MODEL}"

# --- Initialize Tools ---
try:
    news_search_tool = SerperDevTool()
except Exception as e:
    print(f"Warning: Could not initialize SerperDevTool (SERPER_API_KEY might be missing): {e}")
    print("News search functionality will be limited.")
    news_search_tool = None

company_info_tool = CompanyInfoTool()
company_financials_tool = CompanyFinancialsTool()
ticker_search_tool = TickerSearchTool()
historical_stock_data_tool = HistoricalStockDataTool()


# --- Define Agents ---

company_research_agent = Agent(
    role='Lead Company Researcher',
    goal='Gather comprehensive, up-to-date information about a specific publicly traded company using its ticker symbol, including its profile, sector, and key executives.',
    backstory='An expert financial data analyst skilled in sourcing information using precise ticker symbols from financial APIs.',
    verbose=True,
    allow_delegation=False,
    tools=[company_info_tool],
    llm=agent_llm_identifier
)

financial_statement_analyst_agent = Agent(
    role='Quantitative Financial Analyst',
    goal='Analyze financial statement data fetched by tools, calculate key financial ratios (e.g., P/E, Debt-to-Equity, Current Ratio, ROE), and identify significant trends for a given company.',
    backstory='A meticulous analyst with a strong background in accounting and financial modeling, adept at interpreting numbers and financial statements.',
    verbose=True,
    allow_delegation=False,
    tools=[company_financials_tool],
    llm=agent_llm_identifier
)

news_sentiment_analyst_agent = Agent(
    role='Market News Interpreter',
    goal='Find recent news articles and press releases related to a company, summarize them, and gauge their sentiment.',
    backstory='A savvy market watcher who can quickly digest news and understand its potential implications.',
    verbose=True,
    allow_delegation=False,
    tools=[news_search_tool] if news_search_tool else [],
    llm=agent_llm_identifier
)

report_synthesizer_agent = Agent(
    role='Chief Financial Reporter',
    goal='Consolidate findings from all other agents into a coherent, well-structured, and easy-to-understand financial analysis report.',
    backstory='An experienced financial editor with a talent for presenting complex information clearly and concisely.',
    verbose=True,
    allow_delegation=False,
    llm=agent_llm_identifier
)

company_identifier_agent = Agent(
    role='Company Ticker Identifier',
    goal="Identify the correct stock ticker symbol for a company given its name or a partial name. Prioritize matches from major exchanges like NASDAQ, NYSE. If multiple good matches, list the top 2-3.",
    backstory="You are an AI assistant specialized in accurately finding stock ticker symbols. You understand that user input might be ambiguous and aim to provide the most relevant ticker(s).",
    verbose=True,
    allow_delegation=False,
    tools=[ticker_search_tool],
    llm=agent_llm_identifier
)

stock_performance_analyst_agent = Agent(
    role='Stock Performance Analyst',
    goal="Analyze and summarize a company's historical stock performance over the last year using the Historical Stock Data Tool.",
    backstory="You are an analyst specializing in interpreting historical stock price movements to provide a concise overview of recent performance trends.",
    verbose=True,
    allow_delegation=False,
    tools=[historical_stock_data_tool],
    llm=agent_llm_identifier
)

# --- Pydantic Models for Structured Output ---
class FinancialRatio(BaseModel):
    name: str = PydanticField(description="Name of the financial ratio")
    formula: str = PydanticField(description="Formula used for calculation")
    value: Optional[str] = PydanticField(description="Calculated value of the ratio, or 'Cannot calculate due to missing [specific missing data item]'")
    interpretation: str = PydanticField(description="Brief interpretation of the ratio")

class FinancialAnalysisOutput(BaseModel):
    health_summary: str = PydanticField(description="Concise summary of the company's financial health based on the provided key figures, noting any missing data.")
    ratios: List[FinancialRatio] = PydanticField(description="List of calculated financial ratios, each showing: Formula, Calculation with numbers (or placeholders for missing data), and Result (or 'Cannot calculate due to missing [item]').")


# --- Define Tasks ---

# Task for Company Research Agent
company_research_task = Task(
    description="Using the Company Information Tool, fetch the company profile, sector, industry, country, website, business summary, and number of full-time employees for {company_ticker}.",
    expected_output="A structured summary of the company's information including: company_name, sector, industry, country, website, long_business_summary, and full_time_employees.",
    agent=company_research_agent,
)

# Task for Financial Statement Analyst Agent
financial_analysis_task = Task(
    description="""Using the Company Financial Statements Tool, you will receive a summary of the latest annual key figures 
    (like Total Revenue, Net Income, Total Assets, Total Liabilities, Total Equity, Current Assets, Current Liabilities, Operating Cash Flow) for {company_ticker}. 
    Analyze these provided key figures to:
    1. Briefly summarize the company's financial health based on these figures.
    2. Calculate the following financial ratios. For each ratio, explicitly state the formula you are using with the variable names from the provided data (e.g., 'Net Income / Total Revenue').
        Then, show the calculation with the actual numbers (or 'Not available' if a number is missing).
        Finally, provide the result of the ratio or state 'Cannot calculate due to missing [specific missing data item]'.
        - Gross Profit Margin (Gross Profit / Total Revenue)
        - Net Profit Margin (Net Income / Total Revenue)
        - Debt-to-Equity Ratio (Total Liabilities / Total Equity)
        - Current Ratio (Current Assets / Current Liabilities)
    Present the calculated ratios clearly. If a figure needed for a ratio is 'Not available' or None, explicitly state that the ratio cannot be calculated for that reason.""",
    expected_output="""A concise summary of the company's financial health based on the provided key figures.
    Followed by a list of calculated financial ratios, each showing: Formula, Calculation with numbers (or placeholders for missing data), and Result (or 'Cannot calculate due to missing [item]'). Example:
    - Gross Profit Margin: [Value or 'Cannot calculate']
    - Net Profit Margin: [Value or 'Cannot calculate']
    - Debt-to-Equity Ratio: [Value or 'Cannot calculate']
    - Current Ratio: [Value or 'Cannot calculate']
    Provide brief interpretations for each calculated ratio.""",
    output_pydantic=FinancialAnalysisOutput,
    agent=financial_statement_analyst_agent,
    context=[company_research_task] 
)

# Task for News & Sentiment Analyst Agent
news_analysis_task = Task(
    description="""Using the news search tool, find 3-5 recent (last month) **news articles or official press releases** specifically about the company associated with the ticker {company_ticker}. 
    Avoid financial data pages or stock price listings. Focus on textual news content.
    For each article/press release found:
    1. Provide a concise 1-2 sentence summary.
    2. Determine the sentiment (positive, negative, or neutral) based on the news content.
    If no relevant news articles are found, explicitly state that.""",
    expected_output="A list of 3-5 news summaries, each with its determined sentiment. If no news is found, state 'No recent relevant news articles found for {company_ticker}'.",
    agent=news_sentiment_analyst_agent,
    context=[company_research_task]
)

# Task for Stock Performance Analyst Agent
stock_performance_analysis_task = Task(
    description="Using the Historical Stock Data Tool, fetch the 1-year historical stock performance summary for {company_ticker}. Summarize the key findings, including the period, highest price, lowest price, and price at the end of the period.",
    expected_output="A concise summary of the 1-year stock performance, noting the period, highest price, lowest price, and the closing price at the end of the 1-year period.",
    agent=stock_performance_analyst_agent,
    context=[company_research_task]
)

# Task for Report Synthesizer Agent
report_synthesis_task = Task(
    description="Compile all gathered information (company overview, financial analysis & ratios, stock performance summary, and news sentiment) for {company_ticker} into a comprehensive financial report. Structure the report with clear sections: 1. Company Overview, 2. Financial Highlights & Key Ratios, 3. Recent Stock Performance (1-year), 4. Recent News & Sentiment, and 5. Concluding Summary. For the 'Concluding Summary' section, present the key takeaways as a list of 3-5 bullet points. Ensure the overall language is professional and objective.",
    expected_output="A final, well-formatted Markdown report. Section 5, 'Concluding Summary', must be a list of 3-5 bullet points summarizing the key findings. Include a new section 'Recent Stock Performance (1-year)'.",
    agent=report_synthesizer_agent,
    context=[company_research_task, financial_analysis_task, stock_performance_analysis_task, news_analysis_task]
)

# Task for Company Identifier Agent
identify_company_task = Task(
    description="The user has provided the input: '{user_company_input}'. Determine if this is a ticker symbol or a company name. If it seems like a name, use the Ticker Search Tool to find the most relevant stock ticker symbol. If it already looks like a ticker (e.g., 1-5 uppercase letters), assume it's the ticker. Your final output should be ONLY the determined ticker symbol (e.g., AAPL) or a clear message if no suitable ticker is found from search results (e.g., 'Ticker not found for [company_name]'). If multiple good matches, pick the one that seems most prominent or from a major US exchange.",
    expected_output="The single, most relevant stock ticker symbol (e.g., 'MSFT') or a 'Ticker not found...' message.",
    agent=company_identifier_agent
)

# --- Create and Run the Crew ---

financial_analysis_crew = Crew(
    agents=[
        company_identifier_agent,
        company_research_agent, 
        financial_statement_analyst_agent, 
        stock_performance_analyst_agent,
        news_sentiment_analyst_agent, 
        report_synthesizer_agent
    ],
    tasks=[
        identify_company_task,
        company_research_task, 
        financial_analysis_task, 
        stock_performance_analysis_task, 
        news_analysis_task, 
        report_synthesis_task
    ],
    process=Process.sequential,
    verbose=True
)


# --- Streamlit UI ---
st.set_page_config(page_title="AI Financial Analyst", layout="wide")
st.title("📈 AI Financial Analyst Assistant")
st.markdown("Enter a company name or ticker symbol to get a financial analysis report. The more specific the company name, the better the ticker search results.")

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

st.sidebar.header("Company Analysis")
company_input = st.sidebar.text_input(
    "Enter Company Name or Ticker (e.g., Apple or AAPL):", 
    value="AAPL"
)

analyze_button_label = "📊 Analyze Company"
if st.session_state.selected_ticker and st.session_state.get('last_company_input') == company_input:
    analyze_button_label = f"📊 Analyze {st.session_state.selected_ticker}"

if st.sidebar.button(analyze_button_label, key="analyze_button"):
    if not company_input:
        st.warning("⚠️ Please enter a company name or ticker.")
    else:
        if not st.session_state.selected_ticker or st.session_state.get('last_company_input') != company_input:
            st.session_state.selected_ticker = None
            st.session_state.last_company_input = company_input

            with st.spinner(f"🔍 Searching for ticker for '{company_input}'..."):
                try:
                    search_result_data = ticker_search_tool.run(company_name=company_input)

                    if isinstance(search_result_data, dict) and search_result_data.get("error"):
                        st.error(f"API Error during ticker search: {search_result_data['error']}")
                    elif isinstance(search_result_data, dict) and search_result_data.get("api_limit_note"):
                        st.warning(f"Ticker Search API Note: {search_result_data['api_limit_note']}. Results might be incomplete.")
                    elif isinstance(search_result_data, dict) and search_result_data.get("matches"):
                        matches = search_result_data["matches"]
                        if not matches:
                            st.error(f"No ticker symbols found for '{company_input}'. Please try a different name or check spelling.")
                        elif len(matches) == 1:
                            st.session_state.selected_ticker = matches[0].get("1. symbol")
                            st.success(f"Ticker found: {st.session_state.selected_ticker} for {matches[0].get('2. name')}")
                        else:
                            # Present options to the user if multiple matches
                            st.info(f"Multiple potential tickers found for '{company_input}'. Please select one:")
                            match_options = {
                                f"{match.get('1. symbol')} - {match.get('2. name')} ({match.get('4. region')})": match.get("1. symbol")
                                for match in matches[:5] # Show top 5
                            }
                            selected_display_name = st.selectbox("Select the correct company:", options=list(match_options.keys()))
                            if selected_display_name:
                                st.session_state.selected_ticker = match_options[selected_display_name]
                                st.rerun() # Rerun to update button label and proceed

                    else: # Fallback to identifier agent if direct tool use is problematic or for complex cases
                        identifier_inputs = {'user_company_input': company_input}
                        identifier_crew = Crew(agents=[company_identifier_agent], tasks=[identify_company_task], verbose=False)
                        identified_ticker_result = identifier_crew.kickoff(inputs=identifier_inputs)
                        identified_ticker_str = str(identified_ticker_result).strip()

                        if "Ticker not found" in identified_ticker_str or not identified_ticker_str or len(identified_ticker_str) > 7:
                            st.error(f"Could not reliably identify a ticker for '{company_input}'. Agent result: {identified_ticker_str}")
                        else:
                            st.session_state.selected_ticker = identified_ticker_str
                            st.success(f"Ticker identified by agent: {st.session_state.selected_ticker}")
                            st.rerun() # Rerun to update button label

                except Exception as e:
                    st.error(f"❌ An error occurred during ticker identification: {e}")

        # Proceed with analysis if a ticker has been selected/identified
        if st.session_state.selected_ticker:
            st.info(f"🚀 Starting analysis for {st.session_state.selected_ticker} ({company_input})...")
            with st.spinner(f"🤖 Agents are working on {st.session_state.selected_ticker}... This may take a few moments."):
                try:
                    inputs = {'company_ticker': st.session_state.selected_ticker, 'user_company_input': company_input}
                    result = financial_analysis_crew.kickoff(inputs=inputs)
                    
                    st.success(f"✅ Analysis Complete for {st.session_state.selected_ticker}!")
                    st.markdown("---")
                    st.subheader(f"Financial Analysis Report for {st.session_state.selected_ticker} ({company_input})")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ An error occurred during the main crew execution: {e}")
            st.session_state.selected_ticker = None
            st.session_state.last_company_input = None
else:
    st.info("Enter a company name or ticker in the sidebar and click 'Analyze Company' to start.")