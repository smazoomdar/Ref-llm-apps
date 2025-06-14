import gradio as gr
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

# --- Data Schemas ( 그대로 유지 ) ---
@dataclass
class AQIResponse:
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

# --- AQI Data Fetching ( 그대로 유지 ) ---
class AQIAnalyzer:
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API Key is required.")
        self.app = FirecrawlApp(api_key=self.firecrawl_api_key)

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Optional[dict]:
        query = f"Current AQI data for {city}, {state}, {country}"
        search_results = self.app.search(query, page_options={"fetch_page_content": False, "limit": 3})

        if not search_results:
            return None

        # For simplicity, try to scrape the first relevant-looking URL.
        # This part might need refinement based on actual search result quality.
        scrape_url = None
        for result in search_results:
            if "aqicn.org" in result.get("url", "") or "iqair.com" in result.get("url", "") or "airnow.gov" in result.get("url", ""):
                scrape_url = result["url"]
                break

        if not scrape_url and search_results: # Fallback to first result if specific sites not found
            scrape_url = search_results[0]["url"]

        if scrape_url:
            try:
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
            except Exception as e:
                print(f"Error scraping AQI data from {scrape_url}: {e}")
        return None

# --- Modified analyze_conditions function ---
def analyze_conditions(
    openai_api_key: str,
    firecrawl_api_key_input: str,
    city: str,
    state: str,
    country: str,
    medical_conditions: str,
    planned_activity: str
) -> tuple[Optional[str], Optional[str], str, str]:

    warning_message = ""
    info_message = "Processing..."

    if not openai_api_key:
        return None, None, "OpenAI API Key is missing.", "Error"
    if not firecrawl_api_key_input:
        return None, None, "Firecrawl API Key is missing.", "Error"
    if not all([city, state, country]):
        return None, None, "City, State, and Country are required.", "Error"

    os.environ["OPENAI_API_KEY"] = openai_api_key # Set for CrewAI

    user_input_data = UserInput(
        city=city,
        state=state,
        country=country,
        medical_conditions=medical_conditions or "None specified",
        planned_activity=planned_activity or "General daily activities"
    )

    try:
        analyzer = AQIAnalyzer(firecrawl_api_key=firecrawl_api_key_input)
        aqi_data = analyzer.fetch_aqi_data(user_input_data.city, user_input_data.state, user_input_data.country)
    except Exception as e:
        return None, None, f"Error fetching AQI data: {e}", "Error"

    if not aqi_data:
        warning_message = f"Could not retrieve AQI data for {user_input_data.city}. Recommendations will be general."
        # Create a dummy aqi_data structure if none is found to allow agent to still run
        aqi_data = {
            "aqi_value": "Not Available",
            "main_pollutant": "Not Available",
            "health_implications": "AQI data not available, so specific health implications cannot be determined. General caution is advised if you have sensitivities.",
            "cautionary_statements": ["Monitor local alerts if you have health concerns."]
        }

    aqi_json_str = json.dumps(aqi_data, indent=2)

    # --- CrewAI Agent and Task ---
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
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7), # or gpt-3.5-turbo
            verbose=True,
            allow_delegation=False
        )

        recommendation_task = Task(
            description=(
                f"Analyze the provided Air Quality Index (AQI) data and user-specific information to generate personalized health recommendations.\n\n"
                f"Current Location: {user_input_data.city}, {user_input_data.state}, {user_input_data.country}\n"
                f"User's Pre-existing Medical Conditions: {user_input_data.medical_conditions}\n"
                f"User's Planned Outdoor Activity: {user_input_data.planned_activity}\n\n"
                f"Current AQI Data:\n{aqi_json_str}\n\n"
                f"Based on all this information, provide comprehensive health advice. Consider the AQI value, main pollutant, "
                f"stated health implications, and cautionary statements from the data. "
                f"Tailor your advice specifically for the user, considering their medical conditions and planned activities. "
                f"If AQI data is 'Not Available', provide general advice for someone with the user's conditions planning such activities."
                f"Output should be a well-formatted markdown string with actionable recommendations."
            ),
            expected_output=(
                "A detailed markdown string containing personalized health recommendations. This should include: \n"
                "1. A brief summary of the current air quality situation and its relevance to the user.\n"
                "2. Specific advice related to the user's medical conditions (e.g., 'If you have asthma...').\n"
                "3. Recommendations for the planned outdoor activity (e.g., 'Consider rescheduling your run if...', 'If you must go out for your walk...').\n"
                "4. General protective measures if applicable (e.g., 'Wear a mask like N95...', 'Keep windows closed...').\n"
                "5. When to seek medical attention if symptoms worsen."
            ),
            agent=health_advisor_agent
        )

        # Create and run the crew
        aqi_crew = Crew(
            agents=[health_advisor_agent],
            tasks=[recommendation_task],
            process=Process.sequential,
            verbose=True
        )

        # Input for kickoff can be minimal if all data is in task description
        # Or, pass a dictionary if tasks use interpolation for inputs (not strictly needed here as it's all in description)
        crew_result = aqi_crew.kickoff(inputs={
            'location_city': user_input_data.city, # Example of passing inputs, though not directly used by this task desc.
            'user_conditions': user_input_data.medical_conditions
        })

        health_recommendations = crew_result
        info_message = "Analysis complete."

    except Exception as e:
        health_recommendations = f"Error generating health recommendations: {e}"
        info_message = "Error"
        import traceback
        print(traceback.format_exc())


    return aqi_json_str, health_recommendations, info_message, warning_message

# --- Gradio Interface Setup ( 그대로 유지, 함수 호출만 변경 ) ---
openai_api_key_input = gr.Textbox(
    label="OpenAI API Key",
    placeholder="Enter your OpenAI API Key here...",
    type="password",
    lines=1,
    value=os.getenv("OPENAI_API_KEY", "")
)

firecrawl_api_key_gradio_input = gr.Textbox( # Renamed to avoid conflict
    label="Firecrawl API Key",
    placeholder="Enter your Firecrawl API Key here...",
    type="password",
    lines=1,
    value=os.getenv("FIRECRAWL_API_KEY", "")
)

city_input = gr.Textbox(label="City", placeholder="e.g., San Francisco")
state_input = gr.Textbox(label="State/Region", placeholder="e.g., California")
country_input = gr.Textbox(label="Country", placeholder="e.g., USA")
medical_conditions_input = gr.Textbox(label="Pre-existing Medical Conditions (Optional)", placeholder="e.g., Asthma, Allergies")
planned_activity_input = gr.Textbox(label="Planned Outdoor Activity (Optional)", placeholder="e.g., Running, Gardening")

output_aqi_json = gr.JSON(label="Current AQI Data")
output_recommendations = gr.Markdown(label="Personalized Health Recommendations")
output_info = gr.Textbox(label="Status", interactive=False)
output_warning = gr.Textbox(label="Warnings", interactive=False)

# Define the interface
iface = gr.Interface(
    fn=analyze_conditions,
    inputs=[
        openai_api_key_input,
        firecrawl_api_key_gradio_input,
        city_input,
        state_input,
        country_input,
        medical_conditions_input,
        planned_activity_input
    ],
    outputs=[output_aqi_json, output_recommendations, output_info, output_warning],
    title="AI AQI Health Advisor (CrewAI)",
    description="Enter your location and health details to get personalized advice based on current Air Quality Index (AQI) data. This version uses CrewAI for recommendations.",
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()
