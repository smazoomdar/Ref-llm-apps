import streamlit as st
import os
import re
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from firecrawl import FirecrawlApp
from exa_py import Exa
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Schema ---
class CompetitorDataSchema(BaseModel):
    company_name: str = Field(..., description="The name of the company.")
    company_description: Optional[str] = Field(None, description="A brief description of the company and its main products or services.")
    key_features: Optional[List[str]] = Field(default_factory=list, description="List of key features or services offered.")
    pricing_model: Optional[str] = Field(None, description="Description of their pricing model (e.g., subscription, one-time fee, freemium).")
    target_audience: Optional[str] = Field(None, description="The primary target audience of the company.")
    strengths: Optional[List[str]] = Field(default_factory=list, description="Key strengths of the company.")
    weaknesses: Optional[List[str]] = Field(default_factory=list, description="Key weaknesses of the company.")
    website_url: str = Field(..., description="The URL of the company's website.")

# --- API Key Management ---
def get_api_key(service_name: str, session_state):
    env_var = f"{service_name.upper()}_API_KEY"
    if env_var in os.environ:
        return os.environ[env_var]
    return session_state.get(f"{service_name.lower()}_api_key")

# --- Competitor URL Fetching (adapted from original) ---
def get_competitor_urls(company_url: str, company_description: str, exa_api_key: str) -> List[str]:
    """
    Finds competitor URLs using Exa API.
    """
    if not exa_api_key:
        st.error("Exa API Key is required to find competitors.")
        return []

    exa = Exa(api_key=exa_api_key)
    prompt = f"Find 5 direct competitors for a company with website '{company_url}' and description: '{company_description}'. List their primary website URLs."

    try:
        response = exa.search_and_contents(
            prompt,
            num_results=5,
            type="magic",
            # text=True, # Not needed if we directly access result.url
            # highlights=False, # No need for highlights in this context
        )

        urls = []
        if response.results:
            for result in response.results:
                if result.url:
                    urls.append(result.url)

        # Deduplicate and keep up to 5
        unique_urls = list(dict.fromkeys(urls))
        return unique_urls[:5]

    except Exception as e:
        st.error(f"Error finding competitors with Exa: {e}")
        return []

# --- Custom CrewAI Tools ---
class FirecrawlSchemaExtractionTool(BaseTool):
    name: str = "Firecrawl Schema Extraction Tool"
    description: str = "Extracts structured data from a URL based on a Pydantic schema using Firecrawl."

    def _run(self, url: str, schema: BaseModel, firecrawl_api_key: str = None) -> dict:
        if not firecrawl_api_key:
            raise ValueError("Firecrawl API Key is required.")

        app = FirecrawlApp(api_key=firecrawl_api_key)

        try:
            # Convert Pydantic schema to JSON schema for Firecrawl
            json_schema = schema.model_json_schema()
            extracted_data = app.extract(url=url, extraction_schema=json_schema)
            return extracted_data
        except Exception as e:
            return {"error": f"Failed to extract data from {url}: {str(e)}"}


# --- Streamlit UI Setup ---
st.set_page_config(layout="wide")
st.title("AI Competitor Intelligence Agent Team (CrewAI)")

with st.sidebar:
    st.header("API Keys")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state) or "")
    st.session_state.firecrawl_api_key = st.text_input("Firecrawl API Key", type="password", value=get_api_key("FIRECRAWL", st.session_state) or "")
    st.session_state.exa_api_key = st.text_input("Exa API Key", type="password", value=get_api_key("EXA", st.session_state) or "")

st.header("Your Company Information")
user_company_url = st.text_input("Your Company Website URL", placeholder="https://example.com")
user_company_description = st.text_area("Your Company Description", placeholder="Describe your company, its products, and target audience.")

if st.button("Analyze Competitors"):
    openai_api_key = get_api_key("OPENAI", st.session_state)
    firecrawl_api_key = get_api_key("FIRECRAWL", st.session_state)
    exa_api_key = get_api_key("EXA", st.session_state)

    if not all([openai_api_key, firecrawl_api_key, exa_api_key, user_company_url, user_company_description]):
        st.error("Please provide all API keys and your company information.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key # Set for CrewAI
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key # Potentially for tools
        os.environ["EXA_API_KEY"] = exa_api_key # Potentially for tools

        st.info("Finding competitor URLs...")
        competitor_urls = get_competitor_urls(user_company_url, user_company_description, exa_api_key)

        if not competitor_urls:
            st.warning("No competitor URLs found. Please check your company details or Exa API key.")
        else:
            st.success(f"Found {len(competitor_urls)} competitor URLs: {', '.join(competitor_urls)}")

            # --- Define CrewAI Agents ---
            data_extractor_agent = Agent(
                role='Competitor Data Extractor',
                goal=f'Extract detailed information about competitor companies from their websites using the FirecrawlSchemaExtractionTool. Target schema: {CompetitorDataSchema.model_json_schema()}',
                backstory='An AI agent specialized in using Firecrawl to extract structured data from web pages based on a predefined schema. It focuses on accuracy and completeness of the data.',
                tools=[FirecrawlSchemaExtractionTool()],
                verbose=True,
                allow_delegation=False
            )

            comparison_agent = Agent(
                role='Competitor Comparison Analyst',
                goal='Analyze and compare extracted data of competitors against the user\'s company. Highlight key differences, strengths, and weaknesses.',
                backstory='An AI agent skilled in analyzing structured competitor data, performing comparative analysis, and summarizing findings in a clear, actionable format.',
                verbose=True,
                allow_delegation=False
            )

            market_analysis_agent = Agent(
                role='Market Trend Analyst',
                goal='Identify broader market trends, opportunities, and threats based on the collective competitor data and the user\'s company profile.',
                backstory='An AI agent that looks at the bigger picture, synthesizing information from multiple competitors to provide strategic market insights.',
                verbose=True,
                allow_delegation=False
            )

            # --- Define CrewAI Tasks ---
            # For simplicity, we'll create one extraction task per competitor URL.
            # A more advanced setup might involve a manager agent or dynamic task generation.
            extraction_tasks = []
            for url in competitor_urls:
                task = Task(
                    description=f'Extract competitor data from the URL: {url}. Use the FirecrawlSchemaExtractionTool with the CompetitorDataSchema.',
                    expected_output=f'A JSON object containing the extracted data for {url} conforming to CompetitorDataSchema. Include "website_url": "{url}" in the output.',
                    agent=data_extractor_agent,
                    # Pass API key to the tool if it cannot access environment variables directly in CrewAI's tool execution context
                    # This depends on how CrewAI handles tool execution and env vars.
                    # Assuming tool can access env var or it's passed internally by CrewAI if set globally.
                    # Alternatively, tool's _run method needs to fetch it.
                    # For FirecrawlSchemaExtractionTool, we added firecrawl_api_key to _run method.
                    arguments={'url': url, 'schema': CompetitorDataSchema, 'firecrawl_api_key': firecrawl_api_key}
                )
                extraction_tasks.append(task)

            # Placeholder for user company data - in a real scenario, this might also be extracted or input more formally.
            user_company_data_for_context = {
                "company_name": "User's Company (from input)",
                "company_description": user_company_description,
                "website_url": user_company_url
            }

            comparison_task = Task(
                description=f'Compare the user\'s company ({user_company_url}) with the extracted competitor data. Identify key differentiators, feature gaps, and pricing advantages/disadvantages. User company context: {user_company_data_for_context}',
                expected_output='A comparative analysis report summarizing findings for each competitor against the user\'s company. Use tables for feature and pricing comparisons if applicable.',
                agent=comparison_agent,
                context=extraction_tasks # Make sure comparison_agent has access to the output of extraction_tasks
            )

            market_analysis_task = Task(
                description='Based on the user\'s company information and all extracted competitor data, provide a market analysis. This should include overall market positioning, potential opportunities for the user\'s company, and any emerging threats.',
                expected_output='A market analysis report including trends, opportunities, and threats, with actionable insights for the user\'s company.',
                agent=market_analysis_agent,
                context=[comparison_task] + extraction_tasks # Needs outputs from both extraction and comparison
            )

            # --- Define Crew ---
            competitor_crew = Crew(
                agents=[data_extractor_agent, comparison_agent, market_analysis_agent],
                tasks=extraction_tasks + [comparison_task, market_analysis_task], # Ensure tasks are in logical order
                process=Process.sequential, # Sequential for now, can be hierarchical
                verbose=True
            )

            st.info("Kicking off the CrewAI analysis... This may take a few minutes.")

            # Prepare inputs for the crew if tasks need them directly (though often context is used)
            # For this setup, task arguments and context should handle data flow.
            crew_inputs = {
                'user_company_url': user_company_url,
                'user_company_description': user_company_description,
                # 'competitor_urls': competitor_urls # Already used to create tasks
            }

            try:
                crew_result = competitor_crew.kickoff(inputs=crew_inputs)

                st.subheader("Crew Analysis Results")
                st.markdown("---")

                # The `crew_result` structure depends on CrewAI version and setup.
                # Typically, it's the output of the last task or a combined result.
                # We'll try to display the outputs of the comparison and market analysis tasks.

                st.markdown("### Overall Crew Output:")
                st.write(crew_result) # Raw output for inspection

                # To get specific task outputs, you might need to inspect the tasks after kickoff
                # if crew_result doesn't break them down.
                # For now, we assume crew_result contains the final analysis or we can access task outputs.

                # Displaying Comparison Task Output (assuming it's accessible)
                # This part needs refinement based on how CrewAI returns results from multiple tasks.
                # If comparison_task.output is available after kickoff:
                if comparison_task.output:
                     st.markdown("### Competitor Comparison Analysis:")
                     st.markdown(comparison_task.output.exported_output) # or .raw_output
                else:
                     st.warning("Comparison task output not directly available in crew_result. Displaying full result above.")

                # Displaying Market Analysis Task Output
                if market_analysis_task.output:
                    st.markdown("### Market Analysis Report:")
                    st.markdown(market_analysis_task.output.exported_output)
                else:
                    st.warning("Market analysis task output not directly available in crew_result. Displaying full result above.")


            except Exception as e:
                st.error(f"Error during CrewAI execution: {e}")
                # You might want to log the full traceback here for debugging
                st.error(f"OpenAI API Key set: {bool(openai_api_key)}")
                st.error(f"Firecrawl API Key set: {bool(firecrawl_api_key)}")
                st.error(f"Exa API Key set: {bool(exa_api_key)}")

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, Firecrawl, and Exa.")
