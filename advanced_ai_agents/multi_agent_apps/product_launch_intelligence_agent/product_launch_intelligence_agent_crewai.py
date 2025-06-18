import streamlit as st
import os
from dotenv import load_dotenv

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.llms import ChatOpenAI
from crewai_tools import FirecrawlSearchTool # Corrected import if it's directly from crewai_tools
# If FirecrawlSearchTool is not in crewai_tools, it might be a custom tool or from another package.
# Assuming it's available in crewai_tools for now as per instructions.
# If not, it would be: from langchain_community.tools.firecrawl_search import FirecrawlSearchTool
# and then wrapped as a custom tool if needed, or used directly if compatible.
# Given the prompt implies crewai_tools, I'll stick with that.

load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY"
    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- Formatting Instructions (from original helper, adapted) ---
FORMATTING_INSTRUCTIONS = """
The final report should be structured with the following sections:
## Executive Summary
Provide a concise overview of the key findings and insights.

## Detailed Analysis
Present the detailed analysis based on the generated insight bullets. Expand on each point with explanations, examples, and potential implications. Use subheadings for clarity.

## Strategic Recommendations
Offer actionable recommendations based on the analysis. These should be specific and targeted.

## Conclusion
Summarize the main takeaways and overall outlook.

Use markdown for formatting, including headers, subheaders, bullet points, and bold text for emphasis.
Ensure the report is professional, well-organized, and easy to read.
"""

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Product Launch Intelligence (CrewAI)")
st.title("🚀 Product Launch Intelligence Agent (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state))
    st.session_state.firecrawl_api_key = st.text_input("Firecrawl API Key", type="password", value=get_api_key("FIRECRAWL", st.session_state))

st.header("🏢 Company Information")
company_name = st.text_input("Enter Company Name:", placeholder="e.g., Apple, OpenAI, etc.")

# Initialize session state for responses
if 'competitor_response_crewai' not in st.session_state:
    st.session_state.competitor_response_crewai = ""
if 'market_sentiment_response_crewai' not in st.session_state:
    st.session_state.market_sentiment_response_crewai = ""
if 'launch_metrics_response_crewai' not in st.session_state:
    st.session_state.launch_metrics_response_crewai = ""


# --- Analysis Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Competitor Product Launch Analysis", "❤️ Market Sentiment Tracking", "📊 Launch Performance Metrics"])

def run_analysis_crew(agent_config: dict, task1_desc: str, task2_desc: str, company_name_input: str, session_state_key: str):
    openai_key = get_api_key("OPENAI", st.session_state)
    firecrawl_key = get_api_key("FIRECRAWL", st.session_state)

    if not openai_key or not firecrawl_key:
        st.error("OpenAI and Firecrawl API Keys are required.")
        return
    if not company_name_input:
        st.error("Company Name is required.")
        return

    os.environ["OPENAI_API_KEY"] = openai_key
    # FirecrawlSearchTool typically uses FIRECRAWL_API_KEY from env or passed at init
    os.environ["FIRECRAWL_API_KEY"] = firecrawl_key

    try:
        search_tool = FirecrawlSearchTool() # Assumes it picks up FIRECRAWL_API_KEY from env
        # Alternatively: search_tool = FirecrawlSearchTool(api_key=firecrawl_key)
    except Exception as e:
        st.error(f"Error initializing FirecrawlSearchTool: {e}")
        return

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7) # or gpt-3.5-turbo

    analysis_agent = Agent(
        role=agent_config["role"],
        goal=agent_config["goal"],
        backstory=agent_config["backstory"],
        llm=llm,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False
    )

    task_generate_bullets = Task(
        description=task1_desc.format(company_name=company_name_input),
        expected_output="A string containing tagged bullet points (e.g., using XML-like tags like <bullet>Point 1</bullet><bullet>Point 2</bullet> or simple markdown bullets). These bullets should be evidence-based insights derived from web searches if needed.",
        agent=analysis_agent
    )

    task_compile_report = Task(
        description=task2_desc.format(company_name=company_name_input, formatting_instructions=FORMATTING_INSTRUCTIONS),
        expected_output="A fully formatted markdown report based on the insight bullets from the previous task. The report should follow the specified formatting instructions.",
        agent=analysis_agent,
        context=[task_generate_bullets] # Key: pass output of task1 to task2
    )

    crew = Crew(
        agents=[analysis_agent], # Single agent performs both tasks
        tasks=[task_generate_bullets, task_compile_report],
        process=Process.sequential,
        verbose=True
    )

    with st.spinner(f"🤖 AI Agent ({agent_config['role']}) is working on '{company_name_input}'... This may take a few minutes."):
        try:
            result = crew.kickoff(inputs={'company_name': company_name_input}) # Input for the first task
            st.session_state[session_state_key] = result # Result of the last task (report)
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Error during CrewAI execution: {e}")
            import traceback
            st.text(traceback.format_exc())
            st.session_state[session_state_key] = f"Error: {e}"


with tab1:
    st.subheader("Analyze Competitor Product Launch Strategies")
    st.markdown("""
    This agent analyzes recent product launches from the specified company, focusing on competitor strategies, market positioning, and key features.
    It will generate insight bullets and then compile them into a detailed report.
    """)
    if st.button("Analyze Competitor Launches", key="competitor_analyze"):
        agent_config = {
            "role": "Product Launch Strategy Analyst",
            "goal": f"Analyze competitor product launch strategies for {company_name}, focusing on market positioning, key features, and promotional tactics. Generate insights and compile a report.",
            "backstory": "An expert market analyst specializing in deconstructing product launch campaigns to reveal underlying strategies and effectiveness."
        }
        task1_desc = (
            "Generate up to 10-16 evidence-based insight bullets about {company_name}'s most recent product launches (last 1-2 years). "
            "Focus on their product strategy, target audience, key messaging, and market reception. "
            "Use web search extensively. Frame bullets as factual observations or strong hypotheses based on search results."
        )
        task2_desc = (
            "Using ONLY the bullet points from the previous task (provided as context), craft an in-depth competitor launch analysis report for {company_name}. "
            "Expand each bullet point with detailed explanations, potential implications, and examples if applicable. "
            "Follow these formatting instructions strictly:\n{formatting_instructions}"
        )
        run_analysis_crew(agent_config, task1_desc, task2_desc, company_name, "competitor_response_crewai")

    if st.session_state.competitor_response_crewai:
        st.markdown("---")
        st.markdown(st.session_state.competitor_response_crewai)


with tab2:
    st.subheader("Track Market Sentiment for Recent Launches")
    st.markdown("""
    This agent tracks market sentiment related to the company's recent product launches by analyzing online discussions, reviews, and social media reactions.
    It generates insight bullets and then compiles them into a sentiment analysis report.
    """)
    if st.button("Track Market Sentiment", key="sentiment_analyze"):
        agent_config = {
            "role": "Market Sentiment Tracker",
            "goal": f"Analyze and summarize the market sentiment for {company_name}'s recent product launches using web search to find reviews, social media discussions, and news articles. Generate insights and compile a report.",
            "backstory": "A specialist in social listening and sentiment analysis, adept at gauging public opinion and identifying key themes in online conversations about product launches."
        }
        task1_desc = (
            "Generate up to 10-16 evidence-based insight bullets summarizing market sentiment for {company_name}'s recent product launches (last 6-12 months). "
            "Focus on public reception, criticisms, praises, and overall buzz. "
            "Use web search extensively (e.g., search for '{company_name} product reviews', '{company_name} launch reactions')."
        )
        task2_desc = (
            "Using ONLY the bullet points from the previous task (context), create a comprehensive market sentiment report for {company_name}'s recent launches. "
            "Elaborate on each sentiment bullet, providing context and examples if possible. "
            "Follow these formatting instructions strictly:\n{formatting_instructions}"
        )
        run_analysis_crew(agent_config, task1_desc, task2_desc, company_name, "market_sentiment_response_crewai")

    if st.session_state.market_sentiment_response_crewai:
        st.markdown("---")
        st.markdown(st.session_state.market_sentiment_response_crewai)


with tab3:
    st.subheader("Assess Launch Performance Metrics (Qualitative)")
    st.markdown("""
    This agent qualitatively assesses the performance of recent product launches by looking for reports on sales figures (if public), adoption rates, media coverage impact, and overall market influence.
    It generates insight bullets and then compiles them into a performance overview report.
    """)
    if st.button("Assess Launch Metrics", key="metrics_analyze"):
        agent_config = {
            "role": "Launch Performance Metrics Analyst",
            "goal": f"Qualitatively assess the performance of {company_name}'s recent product launches by searching for public data, expert opinions, and media coverage regarding sales, adoption, and market impact. Generate insights and compile a report.",
            "backstory": "An analyst skilled in piecing together qualitative indicators of launch success from public domain information when hard quantitative data is scarce."
        }
        task1_desc = (
            "Generate up to 10-16 evidence-based insight bullets regarding the qualitative performance metrics of {company_name}'s recent product launches (last 1-2 years). "
            "Focus on reported sales data (if available), user adoption trends, media coverage impact, and perceived market influence. "
            "Use web search extensively for articles, reports, and analyses."
        )
        task2_desc = (
            "Using ONLY the bullet points from the previous task (context), compile a qualitative launch performance report for {company_name}. "
            "Expand on each bullet, discussing its implications for the company's success and market standing. "
            "Follow these formatting instructions strictly:\n{formatting_instructions}"
        )
        run_analysis_crew(agent_config, task1_desc, task2_desc, company_name, "launch_metrics_response_crewai")

    if st.session_state.launch_metrics_response_crewai:
        st.markdown("---")
        st.markdown(st.session_state.launch_metrics_response_crewai)

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, and Firecrawl.")
