import streamlit as st
from dataclasses import dataclass, field
from pydantic import BaseModel, Field as PydanticField
from typing import Optional, List
import os
import json
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.llms import ChatOpenAI

load_dotenv()

# --- Data Schemas (Copied from Gradio CrewAI version) ---
@dataclass
class AQIResponseFromScrape: # Renamed to avoid confusion if there's another AQIResponse
    city: str = field(default_factory=str)
    state: str = field(default_factory=str)
    country: str = field(default_factory=str)
    aqi_value: Optional[int] = None
    main_pollutant: Optional[str] = None
    health_implications: Optional[str] = None
    cautionary_statements: Optional[List[str]] = field(default_factory=list)

class ExtractSchema(BaseModel):
    aqi_value: int = PydanticField(..., description="The overall AQI value.")
    main_pollutant: str = PydanticField(..., description="The main pollutant (e.g., PM2.5, O3).")
    health_implications: str = PydanticField(..., description="Brief health implications of the current AQI.")
    cautionary_statements: List[str] = PydanticField(..., description="Specific cautionary advice for different groups.")

@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: str
    planned_activity: str

# --- AQI Data Fetching (Copied from Gradio CrewAI version) ---
class AQIAnalyzer:
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API Key is required for AQIAnalyzer.")
        self.app = FirecrawlApp(api_key=self.firecrawl_api_key)

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Optional[dict]:
        query = f"Current AQI data for {city}, {state}, {country}"
        try:
            search_results = self.app.search(query, page_options={"fetch_page_content": False, "limit": 3})
        except Exception as e:
            st.error(f"Firecrawl search failed: {e}")
            return None

        if not search_results:
            return None

        scrape_url = None
        for result in search_results:
            url = result.get("url", "")
            if any(domain in url for domain in ["aqicn.org", "iqair.com", "airnow.gov"]):
                scrape_url = url
                break

        if not scrape_url and search_results:
            scrape_url = search_results[0]["url"]

        if scrape_url:
            try:
                st.info(f"Attempting to scrape AQI data from: {scrape_url}")
                extracted_data = self.app.scrape_page(
                    scrape_url,
                    params={
                        "pageOptions": {"onlyMainContent": True},
                        "extractorOptions": {
                            "mode": "llm-extraction",
                            "extractionPrompt": f"Extract the AQI value, main pollutant, health implications, and cautionary statements for {city}, {state}, {country}. If multiple AQI values are present (e.g., US AQI, CN AQI), prioritize US AQI or a general one if US AQI is not available.",
                            "extractionSchema": ExtractSchema.model_json_schema()
                        }
                    }
                )
                if extracted_data and extracted_data.get("llm_extraction"):
                    return extracted_data["llm_extraction"]
                else:
                    st.warning(f"LLM extraction failed or returned no data from {scrape_url}. Raw scrape data: {extracted_data}")
                    return None
            except Exception as e:
                st.error(f"Error scraping AQI data from {scrape_url}: {e}")
        return None

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI AQI Health Advisor (Streamlit CrewAI)", layout="wide")
st.title("🌬️ AI AQI Health Advisor (Streamlit & CrewAI)")
st.markdown("Get personalized health recommendations based on current Air Quality Index (AQI) data.")

# Sidebar for API key configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key", ""),
        help="Enter your OpenAI API key."
    )
    st.session_state.firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        type="password",
        value=os.getenv("FIRECRAWL_API_KEY") or st.session_state.get("firecrawl_api_key", ""),
        help="Enter your Firecrawl API key for fetching AQI data."
    )

# Main UI for inputs
st.subheader("📍 Your Location")
col1, col2, col3 = st.columns(3)
with col1:
    city = st.text_input("City", placeholder="e.g., San Francisco")
with col2:
    state = st.text_input("State/Region", placeholder="e.g., California (optional for some countries)")
with col3:
    country = st.text_input("Country", placeholder="e.g., USA")

st.subheader("❤️ Your Health & Activity Profile")
medical_conditions = st.text_area(
    "Pre-existing Medical Conditions (Optional)",
    placeholder="e.g., Asthma, Allergies, Cardiovascular issues. Leave blank if none."
)
planned_activity = st.text_area(
    "Planned Outdoor Activity (Optional)",
    placeholder="e.g., Morning run for 1 hour, Gardening, Attending an outdoor event. Leave blank if none specific."
)

if st.button("Analyze & Get Recommendations", type="primary"):
    openai_key = st.session_state.get("openai_api_key")
    firecrawl_key = st.session_state.get("firecrawl_api_key")

    if not openai_key:
        st.error("❌ OpenAI API Key is missing. Please enter it in the sidebar.")
    elif not firecrawl_key:
        st.error("❌ Firecrawl API Key is missing. Please enter it in the sidebar.")
    elif not all([city, country]): # State is optional for some countries
        st.error("❌ City and Country are required fields.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key # Set for CrewAI

        user_input = UserInput(
            city=city,
            state=state or "N/A", # Use N/A if state is empty
            country=country,
            medical_conditions=medical_conditions or "None specified",
            planned_activity=planned_activity or "General daily activities"
        )

        st.info(f"Fetching AQI data for {user_input.city}, {user_input.state}, {user_input.country}...")

        aqi_data = None
        warning_message = ""
        try:
            analyzer = AQIAnalyzer(firecrawl_api_key=firecrawl_key)
            aqi_data = analyzer.fetch_aqi_data(user_input.city, user_input.state, user_input.country)
        except ValueError as ve: # Catch API key error from AQIAnalyzer init
             st.error(str(ve))
             st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while initializing AQI Analyzer or fetching data: {e}")
            # Proceed with dummy data for agent if AQI fetch fails, so user still gets general advice
            warning_message = f"Could not retrieve live AQI data due to an error: {e}. Recommendations will be general."

        if not aqi_data:
            if not warning_message: # If no specific error message yet
                 warning_message = f"Could not retrieve AQI data for {user_input.city}. Recommendations will be general."
            aqi_data_for_agent = { # Use this structure for the agent if live data fails
                "aqi_value": "Not Available",
                "main_pollutant": "Not Available",
                "health_implications": "Live AQI data not available for your location. Specific health implications cannot be determined. General caution is advised, especially if you have sensitivities.",
                "cautionary_statements": ["Monitor local alerts and your personal health if you have concerns."]
            }
            st.warning(warning_message)
        else:
            aqi_data_for_agent = aqi_data
            st.success(f"Successfully fetched AQI data for {user_input.city}.")


        aqi_json_display = json.dumps(aqi_data_for_agent, indent=2)

        st.subheader("📊 Current AQI Data (or best available)")
        st.json(aqi_json_display)

        with st.spinner("🤖 AI Health Advisor is analyzing and generating recommendations..."):
            try:
                health_advisor_agent = Agent(
                    role="Expert Health Advisor specializing in Air Quality Impacts",
                    goal=(
                        "Provide personalized and actionable health recommendations based on the current Air Quality Index (AQI) data, "
                        "the user's specified medical conditions, and their planned outdoor activities. "
                        "The recommendations should be clear, concise, and tailored to minimize health risks associated with air pollution."
                    ),
                    backstory=(
                        "You are a highly respected public health expert with years of experience in environmental health and toxicology. "
                        "You have a deep understanding of how different air pollutants affect human health, particularly vulnerable populations. "
                        "Your advice is sought after for its practicality and evidence-based approach. You always aim to empower individuals "
                        "to protect their health without causing undue alarm."
                    ),
                    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7),
                    verbose=True,
                    allow_delegation=False
                )

                recommendation_task = Task(
                    description=(
                        f"Analyze the provided Air Quality Index (AQI) data and user-specific information to generate personalized health recommendations.\n\n"
                        f"Current Location: {user_input.city}, {user_input.state}, {user_input.country}\n"
                        f"User's Pre-existing Medical Conditions: {user_input.medical_conditions}\n"
                        f"User's Planned Outdoor Activity: {user_input.planned_activity}\n\n"
                        f"Current AQI Data (JSON format):\n{json.dumps(aqi_data_for_agent)}\n\n" # Pass as JSON string in description
                        f"Based on all this information, provide comprehensive health advice. Consider the AQI value, main pollutant, "
                        f"stated health implications, and cautionary statements from the data. "
                        f"Tailor your advice specifically for the user, considering their medical conditions and planned activities. "
                        f"If AQI data indicates 'Not Available', provide general advice for someone with the user's conditions planning such activities, assuming potentially poor air quality as a precaution."
                        f"Format your output as a well-structured markdown string with clear sections and actionable recommendations."
                    ),
                    expected_output=(
                        "A detailed markdown string containing personalized health recommendations. This should include sections such as: \n"
                        "1. **Air Quality Summary**: Briefly interpret the provided AQI data for the user.\n"
                        "2. **Impact on Your Health**: Specifically address how the current (or assumed poor if data is unavailable) air quality might affect someone with the user's stated medical conditions.\n"
                        "3. **Advice for Planned Activity**: Recommendations regarding the safety and timing of the planned outdoor activity.\n"
                        "4. **General Protective Measures**: Suggest general actions like using masks, air purifiers, etc., if appropriate for the AQI level or lack of data.\n"
                        "5. **When to Be Concerned**: Symptoms or situations that should prompt the user to seek medical advice or take extra precautions."
                    ),
                    agent=health_advisor_agent
                )

                aqi_crew = Crew(
                    agents=[health_advisor_agent],
                    tasks=[recommendation_task],
                    process=Process.sequential,
                    verbose=False # Set to True for more detailed logs in terminal
                )

                crew_result = aqi_crew.kickoff(inputs={ # Inputs for kickoff, if task uses {{placeholder}}
                    'aqi_data_json': json.dumps(aqi_data_for_agent), # Example if task used {{aqi_data_json}}
                    'user_medical_conditions': user_input.medical_conditions,
                    'user_planned_activity': user_input.planned_activity,
                    'user_location': f"{user_input.city}, {user_input.state}, {user_input.country}"
                })

                st.subheader("🩺 Personalized Health Recommendations")
                st.markdown(crew_result)
                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Error generating health recommendations via CrewAI: {e}")
                import traceback
                st.text(traceback.format_exc())

st.markdown("---")
st.caption("Disclaimer: This tool provides AI-generated advice and does not substitute professional medical consultation. Always consult with a healthcare provider for medical concerns. AQI data accuracy depends on the source (Firecrawl).")
