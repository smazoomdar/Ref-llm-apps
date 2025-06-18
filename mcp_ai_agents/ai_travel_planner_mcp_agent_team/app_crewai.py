import streamlit as st
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool

# Google Calendar specific imports
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow # For refresh token, if needed
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, is_google_cred: bool = False):
    if is_google_cred:
        prefix = f"google_{service_name.lower()}"
    else:
        prefix = service_name.lower()

    env_var = f"{prefix.upper()}_API_KEY" if not is_google_cred else prefix.upper()

    if service_name == "google_client_id": env_var = "GOOGLE_CLIENT_ID"
    if service_name == "google_client_secret": env_var = "GOOGLE_CLIENT_SECRET"
    if service_name == "google_refresh_token": env_var = "GOOGLE_REFRESH_TOKEN"

    if env_var in os.environ:
        return os.environ[env_var]
    return session_state.get(f"{prefix}_api_key")


# --- Custom CrewAI Tools ---

# Placeholder Tools
class PlaceholderAirbnbTool(BaseTool):
    name: str = "Placeholder Airbnb Tool"
    description: str = "Simulates searching for accommodations on Airbnb."

    def _run(self, criteria: str) -> str:
        return f"Success: Airbnb search for '{criteria}' would be performed here. (No actual API call)"

class PlaceholderGoogleMapsTool(BaseTool):
    name: str = "Placeholder Google Maps Tool"
    description: str = "Simulates fetching data from Google Maps."

    def _run(self, location_query: str) -> str:
        return f"Success: Google Maps data for '{location_query}' would be retrieved here. (No actual API call)"

class PlaceholderWeatherTool(BaseTool):
    name: str = "Placeholder Weather Tool"
    description: str = "Simulates fetching weather forecasts."

    def _run(self, location: str, date: str) -> str:
        return f"Success: Weather forecast for {location} on {date} would be fetched here. (No actual API call)"

class CreateCalendarEventTool(BaseTool):
    name: str = "Google Calendar Event Creator"
    description: str = "Creates an event in Google Calendar. Requires Google Client ID, Client Secret, and Refresh Token."

    google_client_id: str = None
    google_client_secret: str = None
    google_refresh_token: str = None

    def __init__(self, google_client_id: str, google_client_secret: str, google_refresh_token: str, **kwargs):
        super().__init__(**kwargs)
        self.google_client_id = google_client_id
        self.google_client_secret = google_client_secret
        self.google_refresh_token = google_refresh_token

    def _run(self, summary: str, start_datetime_str: str, end_datetime_str: str, description: str = None, location: str = None, timezone: str = "UTC") -> str:
        if not all([self.google_client_id, self.google_client_secret, self.google_refresh_token]):
            return "Error: Google API credentials (Client ID, Client Secret, Refresh Token) are missing for Calendar tool."

        creds = Credentials.from_authorized_user_info(
            info={
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "refresh_token": self.google_refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token", # Default token URI
            },
            scopes=["https://www.googleapis.com/auth/calendar.events"]
        )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    return f"Error refreshing Google Calendar token: {e}"
            else:
                return "Error: Could not obtain valid Google Calendar credentials. Please check your setup."

        try:
            service = build("calendar", "v3", credentials=creds)
            event = {
                "summary": summary,
                "location": location,
                "description": description,
                "start": {"dateTime": start_datetime_str, "timeZone": timezone},
                "end": {"dateTime": end_datetime_str, "timeZone": timezone},
            }
            created_event = service.events().insert(calendarId="primary", body=event).execute()
            return f"Success: Event created: {created_event.get('htmlLink')}"
        except HttpError as error:
            return f"Error creating Google Calendar event: {error}"
        except Exception as e:
            return f"An unexpected error occurred with Google Calendar: {e}"

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Travel Planner (CrewAI)")
st.title("✈️ AI Travel Planner (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state) or ""
    )
    st.subheader("Google Calendar Credentials")
    st.caption("Needed for the Itinerary Manager Agent to create calendar events.")
    st.session_state.google_client_id_api_key = st.text_input( # Using _api_key suffix for consistency with helper
        "Google Client ID", type="password", value=get_api_key("google_client_id", st.session_state, is_google_cred=True) or ""
    )
    st.session_state.google_client_secret_api_key = st.text_input(
        "Google Client Secret", type="password", value=get_api_key("google_client_secret", st.session_state, is_google_cred=True) or ""
    )
    st.session_state.google_refresh_token_api_key = st.text_input(
        "Google Refresh Token", type="password", value=get_api_key("google_refresh_token", st.session_state, is_google_cred=True) or ""
    )

st.header("Your Travel Details")
destination = st.text_input("Destination:", placeholder="e.g., Paris, France")
start_date = st.date_input("Start Date:")
end_date = st.date_input("End Date:")
travelers = st.number_input("Number of Travelers:", min_value=1, value=1)
interests = st.text_area("Interests & Preferences:", placeholder="e.g., historical sites, local cuisine, hiking, museums")
budget = st.selectbox("Budget Range:", ["Economy", "Mid-Range", "Luxury"], index=1)

if st.button("Plan My Trip with CrewAI"):
    openai_api_key = get_api_key("OPENAI", st.session_state)
    google_client_id = get_api_key("google_client_id", st.session_state, is_google_cred=True)
    google_client_secret = get_api_key("google_client_secret", st.session_state, is_google_cred=True)
    google_refresh_token = get_api_key("google_refresh_token", st.session_state, is_google_cred=True)

    if not openai_api_key:
        st.error("OpenAI API Key is required.")
    elif not all([destination, start_date, end_date, interests]):
        st.error("Please fill in all travel details: Destination, Start Date, End Date, and Interests.")
    elif not all([google_client_id, google_client_secret, google_refresh_token]):
        st.warning("Google Calendar credentials are not fully provided. The Itinerary Manager may not be able to create events.")
        # Proceed without calendar functionality or make it optional for the agent
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key # Set for CrewAI

        # Instantiate tools that need API keys/credentials
        calendar_tool = CreateCalendarEventTool(
            google_client_id=google_client_id,
            google_client_secret=google_client_secret,
            google_refresh_token=google_refresh_token
        )
        airbnb_tool = PlaceholderAirbnbTool()
        maps_tool = PlaceholderGoogleMapsTool()
        weather_tool = PlaceholderWeatherTool()

        # --- Define CrewAI Agents ---
        location_scout = Agent(
            role='Expert Location Scout',
            goal=f'Find key points of interest, best routes, and local tips for {destination} based on user interests: {interests}.',
            backstory='You are a seasoned travel expert with a knack for uncovering hidden gems and optimizing travel logistics using map data.',
            tools=[maps_tool],
            verbose=True
        )
        weather_forecaster = Agent(
            role='Weather Forecaster Specialist',
            goal=f'Provide a reliable weather forecast for {destination} between {start_date} and {end_date}.',
            backstory='You are an AI that provides accurate, up-to-date weather forecasts crucial for travel planning.',
            tools=[weather_tool],
            verbose=True
        )
        accommodation_finder = Agent(
            role='Accommodation Specialist',
            goal=f'Find suitable accommodation options in {destination} for {travelers} traveler(s) within a {budget} budget, considering these interests: {interests}.',
            backstory='You are an expert in finding the best places to stay, matching traveler preferences with available options.',
            tools=[airbnb_tool],
            verbose=True
        )
        itinerary_manager = Agent(
            role='Master Itinerary Planner & Calendar Organizer',
            goal=f'Create a comprehensive travel itinerary for a trip to {destination} from {start_date} to {end_date} for {travelers} person(s) with interests in {interests} and a {budget} budget. Then, create a single summary calendar event for the trip.',
            backstory='You meticulously plan travel itineraries and ensure they are well-documented and scheduled. You are also proficient in using Google Calendar.',
            tools=[calendar_tool],
            verbose=True
        )

        # --- Define CrewAI Tasks ---
        trip_start_datetime_str = datetime.combine(start_date, datetime.min.time()).isoformat()
        trip_end_datetime_str = datetime.combine(end_date, datetime.max.time()).isoformat()

        task_location_scouting = Task(
            description=f'Scout {destination} for points of interest, activities, and dining relevant to: {interests}. Also, suggest efficient travel routes between them.',
            expected_output=f'A list of recommended locations, activities, and dining options in {destination}, with brief descriptions and suggested routes or travel tips.',
            agent=location_scout
        )
        task_weather_forecast = Task(
            description=f'Get the weather forecast for {destination} for the period from {start_date} to {end_date}.',
            expected_output=f'A weather summary for {destination} covering the dates {start_date} to {end_date}, highlighting any potential concerns for a traveler interested in {interests}.',
            agent=weather_forecaster,
            arguments={'location': destination, 'date': f'{start_date} to {end_date}'} # For PlaceholderWeatherTool
        )
        task_accommodation_search = Task(
            description=f'Find accommodation in {destination} for {travelers} traveler(s). Budget: {budget}. Preferences: {interests}. Dates: {start_date} to {end_date}.',
            expected_output=f'A list of 2-3 suitable accommodation options in {destination} with brief descriptions, price indications, and why they fit the criteria.',
            agent=accommodation_finder,
            arguments={'criteria': f'{travelers} travelers, {destination}, from {start_date} to {end_date}, budget: {budget}, interests: {interests}'} # For PlaceholderAirbnbTool
        )
        task_create_itinerary_and_event = Task(
            description=f'Compile all gathered information (locations, weather, accommodation) into a coherent daily itinerary for the trip to {destination} from {start_date} to {end_date}. Then, create a single Google Calendar event summarizing the trip.',
            expected_output=f'A detailed day-by-day travel plan for {destination}, and a confirmation message from Google Calendar (success or error). The itinerary should be easy to read and follow.',
            agent=itinerary_manager,
            context=[task_location_scouting, task_weather_forecast, task_accommodation_search], # Depends on other tasks
            arguments={ # For CreateCalendarEventTool
                'summary': f'Trip to {destination}',
                'start_datetime_str': trip_start_datetime_str,
                'end_datetime_str': trip_end_datetime_str,
                'description': f'Travel plan for {travelers} person(s) to {destination} with interests: {interests}. Budget: {budget}. More details in the main itinerary.',
                'location': destination,
                'timezone': 'UTC' # Or derive from destination if possible/needed
            }
        )

        # --- Define Crew ---
        travel_crew = Crew(
            agents=[location_scout, weather_forecaster, accommodation_finder, itinerary_manager],
            tasks=[task_location_scouting, task_weather_forecast, task_accommodation_search, task_create_itinerary_and_event],
            process=Process.sequential,
            verbose=True
        )

        st.info("🤖 CrewAI is planning your trip... This might take a moment.")

        crew_inputs = { # These inputs are used by agents/tasks via string formatting in their definitions
            'destination': destination,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'travelers': travelers,
            'interests': interests,
            'budget': budget
        }

        try:
            crew_result = travel_crew.kickoff(inputs=crew_inputs)

            st.subheader("🌍 Your CrewAI Travel Plan:")
            st.markdown("---")
            # The result structure depends on CrewAI. It's often the output of the last task.
            st.markdown(crew_result)

            # Explicitly show calendar event creation result if available in the task output
            calendar_event_output = task_create_itinerary_and_event.output.exported_output if task_create_itinerary_and_event.output else "Calendar event status not directly available."
            if "Success: Event created" in calendar_event_output:
                st.success(f"Calendar Event: {calendar_event_output}")
            elif "Error:" in calendar_event_output:
                 st.error(f"Calendar Event: {calendar_event_output}")
            else: # Fallback if structure is different or no specific message found
                st.info(f"Itinerary Manager Agent Output (includes calendar status): {calendar_event_output}")


        except Exception as e:
            st.error(f"Error during CrewAI trip planning: {e}")
            st.error(f"OpenAI API Key set: {bool(openai_api_key)}")
            st.error(f"Google Client ID set: {bool(google_client_id)}")
            st.error(f"Google Client Secret set: {bool(google_client_secret)}")
            st.error(f"Google Refresh Token set: {bool(google_refresh_token)}")

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, and various (simulated) travel APIs.")
