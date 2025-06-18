import streamlit as st
import os
import smtplib
import requests
import json
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import PyPDF2
# typing imports moved to the top section
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool

load_dotenv()

# --- Helper for API Keys/Credentials ---
def get_config_value(config_key: str, session_state, default_value: str = "", is_sensitive: bool = True):
    env_var_name = config_key.upper()
    if env_var_name in os.environ:
        return os.environ[env_var_name]

    # For Streamlit inputs, keys are usually lowercase
    st_key = config_key.lower()
    if is_sensitive and not session_state.get(f"{st_key}_display", False): # Avoid showing sensitive data from env in input if not explicitly shown
         return session_state.get(st_key, default_value) if session_state.get(f"{st_key}_display") else default_value
    return session_state.get(st_key, default_value)


# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# --- Custom CrewAI Tools ---

class SendEmailTool(BaseTool):
    name: str = "Send Email Tool"
    description: str = "Sends an email to a specified recipient."

    sender_email: str
    sender_password: str
    smtp_server: str
    smtp_port: int

    def __init__(self, sender_email: str, sender_password: str, smtp_server: str, smtp_port: int, **kwargs):
        super().__init__(**kwargs)
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def _run(self, to_email: str, subject: str, body: str) -> str:
        if not all([self.sender_email, self.sender_password, self.smtp_server, self.smtp_port]):
            return "Error: Email sender credentials or server configuration is missing."
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = to_email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, to_email, msg.as_string())
            return f"Success: Email sent to {to_email} with subject '{subject}'."
        except Exception as e:
            return f"Error sending email: {e}"


class ScheduleZoomMeetingTool(BaseTool):
    name: str = "Schedule Zoom Meeting Tool"
    description: str = "Schedules a Zoom meeting with specified attendees."

    zoom_account_id: str
    zoom_client_id: str
    zoom_client_secret: str

    _access_token: Optional[str] = None
    _token_expires_at: Optional[datetime] = None

    def __init__(self, zoom_account_id: str, zoom_client_id: str, zoom_client_secret: str, **kwargs):
        super().__init__(**kwargs)
        self.zoom_account_id = zoom_account_id
        self.zoom_client_id = zoom_client_id
        self.zoom_client_secret = zoom_client_secret

    def _get_access_token(self) -> Optional[str]:
        if self._access_token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._access_token

        auth_url = "https://zoom.us/oauth/token"
        auth_payload = {
            "grant_type": "account_credentials",
            "account_id": self.zoom_account_id,
        }
        try:
            response = requests.post(auth_url, auth=(self.zoom_client_id, self.zoom_client_secret), data=auth_payload)
            response.raise_for_status()
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3000) # Default to 50 mins
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 600) # Refresh 10 mins before expiry
            return self._access_token
        except requests.exceptions.RequestException as e:
            print(f"Error getting Zoom access token: {e}")
            return None

    def _run(self, topic: str, start_time_iso: str, duration_minutes: int, attendee_emails: List[str]) -> str:
        if not all([self.zoom_account_id, self.zoom_client_id, self.zoom_client_secret]):
            return "Error: Zoom API credentials (Account ID, Client ID, Client Secret) are missing."

        access_token = self._get_access_token()
        if not access_token:
            return "Error: Could not obtain Zoom access token."

        meetings_url = f"https://api.zoom.us/v2/users/me/meetings" # 'me' keyword works for user-level apps
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        meeting_payload = {
            "topic": topic,
            "type": 2,  # Scheduled meeting
            "start_time": start_time_iso, # Format: "YYYY-MM-DDTHH:MM:SSZ" or "YYYY-MM-DDTHH:MM:SS" for local time
            "duration": duration_minutes,
            "timezone": "UTC", # Or use a specific timezone
            "settings": {
                "join_before_host": True,
                "mute_upon_entry": True,
                "participant_video": True,
                "host_video": True,
                "auto_recording": "none", # "local", "cloud"
                "alternative_hosts": "", # Comma-separated emails if any
            },
            # Zoom API for creating meetings for users typically doesn't take attendee_emails directly in this payload.
            # Instead, the join_url is sent to attendees.
            # If you need to register attendees, you'd use a separate endpoint.
        }
        if attendee_emails: # For logging or if API changes
             meeting_payload.setdefault("agenda", f"Meeting with {', '.join(attendee_emails)}")


        try:
            response = requests.post(meetings_url, headers=headers, json=meeting_payload)
            response.raise_for_status()
            meeting_data = response.json()
            join_url = meeting_data.get("join_url")
            meeting_id = meeting_data.get("id")
            return f"Success: Zoom meeting scheduled. ID: {meeting_id}, Join URL: {join_url}"
        except requests.exceptions.RequestException as e:
            error_details = e.response.json() if e.response else str(e)
            return f"Error scheduling Zoom meeting: {error_details}"
        except Exception as e:
            return f"An unexpected error occurred with Zoom: {e}"


# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Recruitment Team (CrewAI)")
st.title("🎯 AI Recruitment Agent Team (CrewAI Version)")

with st.sidebar:
    st.header("API & Service Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_config_value("OPENAI_API_KEY", st.session_state))

    st.subheader("Email (SMTP)")
    st.session_state.sender_email = st.text_input("Sender Email Address", value=get_config_value("SENDER_EMAIL", st.session_state, is_sensitive=False))
    st.session_state.sender_password = st.text_input("Sender Email Password", type="password", value=get_config_value("SENDER_PASSWORD", st.session_state))
    st.session_state.smtp_server = st.text_input("SMTP Server", value=get_config_value("SMTP_SERVER", st.session_state, is_sensitive=False))
    st.session_state.smtp_port = st.number_input("SMTP Port", value=int(get_config_value("SMTP_PORT", st.session_state, "587", is_sensitive=False)), min_value=1, max_value=65535)

    st.subheader("Zoom (Server-to-Server OAuth)")
    st.session_state.zoom_account_id = st.text_input("Zoom Account ID", value=get_config_value("ZOOM_ACCOUNT_ID", st.session_state, is_sensitive=False))
    st.session_state.zoom_client_id = st.text_input("Zoom Client ID", value=get_config_value("ZOOM_CLIENT_ID", st.session_state, is_sensitive=False))
    st.session_state.zoom_client_secret = st.text_input("Zoom Client Secret", type="password", value=get_config_value("ZOOM_CLIENT_SECRET", st.session_state))

st.header("📄 Candidate Application")
job_role = st.text_input("Job Role Applied For:", placeholder="e.g., Senior Software Engineer")
candidate_name = st.text_input("Candidate Name:", placeholder="e.g., Jane Doe")
candidate_email = st.text_input("Candidate Email:", placeholder="e.g., jane.doe@example.com")
uploaded_resume = st.file_uploader("Upload Candidate's Resume (PDF)", type="pdf")

# Store screening results in session state
if 'screening_result' not in st.session_state:
    st.session_state.screening_result = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None


if uploaded_resume is not None and st.button("Analyze Resume"):
    st.session_state.resume_text = extract_text_from_pdf(uploaded_resume)
    openai_api_key = get_config_value("OPENAI_API_KEY", st.session_state)

    if not openai_api_key:
        st.error("OpenAI API Key is required for resume screening.")
    elif not st.session_state.resume_text:
        st.error("Could not extract text from the resume.")
    elif not job_role:
        st.error("Please specify the Job Role.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        resume_screener_agent = Agent(
            role='Resume Screener',
            goal=f'Analyze the provided resume against the requirements for a {job_role}. Determine if the candidate is a good fit, highlighting key skills, experience, and any red flags.',
            backstory='An expert HR professional with years of experience in technical recruitment, known for a keen eye in identifying top talent from resumes.',
            verbose=True,
            allow_delegation=False # No delegation for initial screening
        )
        screening_task = Task(
            description=f'Screen the following resume text for the "{job_role}" position:\n\n{st.session_state.resume_text}\n\nProvide a concise summary of the candidate\'s suitability, relevant skills, years of experience for similar roles, and a recommendation (e.g., "Selected for Interview", "Not a good fit", "Potentially a fit with concerns").',
            expected_output='A JSON object containing: "candidate_name_from_resume" (if identifiable, else "N/A"), "suitability_summary", "key_skills", "experience_summary", "recommendation" (string), and "feedback_for_candidate" (constructive, if not selected).',
            agent=resume_screener_agent
        )

        # For now, just run the screening task directly (not a full crew yet for this part)
        # This is a simplified approach for the initial screening.
        # A Crew can be used here too if more complex screening is needed.
        st.info("🤖 Resume Screener Agent is analyzing...")
        try:
            # This is a conceptual direct run, CrewAI tasks usually run within a Crew.
            # For a single agent task, you might define a simple crew or use the agent's execute_task method if available.
            # Let's create a temporary mini-crew for this.
            temp_crew = Crew(agents=[resume_screener_agent], tasks=[screening_task], process=Process.sequential, verbose=0)
            result = temp_crew.kickoff()
            st.session_state.screening_result = result # The raw output from the task

            st.subheader("Screening Result:")
            st.markdown(st.session_state.screening_result) # Display the raw JSON-like string.
            # In a real app, parse this JSON and display nicely.

        except Exception as e:
            st.error(f"Error during resume screening: {e}")
            st.session_state.screening_result = None

if st.session_state.screening_result and candidate_email: # Check if screening was done and email is available
    # Assuming the screening result string contains a positive recommendation to proceed
    # This logic needs to be more robust by parsing the screening_result
    if "Selected for Interview" in st.session_state.screening_result or \
       "Potentially a fit" in st.session_state.screening_result: # Example condition

        st.markdown("---")
        st.header("✉️ Next Steps: Contact Candidate & Schedule Interview")

        # Allow user to set interview date/time
        interview_date = st.date_input("Proposed Interview Date", value=datetime.now().date() + timedelta(days=3))
        interview_time = st.time_input("Proposed Interview Time", value=datetime.strptime("10:00", "%H:%M").time())
        interview_datetime = datetime.combine(interview_date, interview_time)
        interview_datetime_iso = interview_datetime.isoformat() # Zoom might need "Z" for UTC

        # Use a more robust way to get candidate name from screening_result if available
        # For now, using the input field if screening didn't reliably extract it.
        actual_candidate_name = candidate_name # Fallback to input name

        if st.button("Proceed with Application (Send Email & Schedule Interview)"):
            # Get all necessary credentials
            openai_api_key = get_config_value("OPENAI_API_KEY", st.session_state)
            sender_email_val = get_config_value("SENDER_EMAIL", st.session_state, is_sensitive=False)
            sender_password_val = get_config_value("SENDER_PASSWORD", st.session_state)
            smtp_server_val = get_config_value("SMTP_SERVER", st.session_state, is_sensitive=False)
            smtp_port_val = int(get_config_value("SMTP_PORT", st.session_state, "587", is_sensitive=False))

            zoom_account_id_val = get_config_value("ZOOM_ACCOUNT_ID", st.session_state, is_sensitive=False)
            zoom_client_id_val = get_config_value("ZOOM_CLIENT_ID", st.session_state, is_sensitive=False)
            zoom_client_secret_val = get_config_value("ZOOM_CLIENT_SECRET", st.session_state)

            # Validate all credentials before proceeding
            if not all([openai_api_key, sender_email_val, sender_password_val, smtp_server_val,
                        zoom_account_id_val, zoom_client_id_val, zoom_client_secret_val]):
                st.error("Missing one or more API/service credentials in the sidebar for Email or Zoom.")
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key

                # Instantiate Tools
                email_tool = SendEmailTool(
                    sender_email=sender_email_val, sender_password=sender_password_val,
                    smtp_server=smtp_server_val, smtp_port=smtp_port_val
                )
                zoom_tool = ScheduleZoomMeetingTool(
                    zoom_account_id=zoom_account_id_val,
                    zoom_client_id=zoom_client_id_val,
                    zoom_client_secret=zoom_client_secret_val
                )

                # Define Agents
                recruitment_coordinator = Agent(
                    role='Recruitment Coordinator',
                    goal=f'Communicate with candidate {actual_candidate_name} ({candidate_email}) about their application for {job_role}, and inform them about next steps including a scheduled interview.',
                    backstory='An efficient HR coordinator responsible for smooth candidate communication and interview logistics.',
                    tools=[email_tool],
                    verbose=True
                )
                interview_scheduler = Agent(
                    role='Interview Scheduler',
                    goal=f'Schedule a new interview for {actual_candidate_name} for the {job_role} position. The interview should be on {interview_datetime_iso} and last for 45 minutes.',
                    backstory='A specialist in managing interview calendars and using scheduling tools like Zoom effectively.',
                    tools=[zoom_tool],
                    verbose=True
                )

                # Define Tasks
                task_schedule_interview = Task(
                    description=f'Schedule a Zoom interview for candidate {actual_candidate_name} for the role of {job_role}. Interview date and time: {interview_datetime_iso}. Duration: 45 minutes. Attendees: {candidate_email} (and interviewer if known).',
                    expected_output='A confirmation string containing the Zoom meeting ID and Join URL.',
                    agent=interview_scheduler,
                    arguments={ # These are passed to the tool's _run method if the agent decides to use it
                        'topic': f'Interview: {actual_candidate_name} for {job_role}',
                        'start_time_iso': interview_datetime_iso,
                        'duration_minutes': 45,
                        'attendee_emails': [candidate_email] # Add interviewer email if available
                    }
                )

                # This task's description will use the output of task_schedule_interview
                task_send_interview_email = Task(
                    description=f'Compose and send an email to {actual_candidate_name} ({candidate_email}) regarding their application for {job_role}. Inform them they have been selected for an interview. Include the Zoom meeting details from the scheduled interview (context from task_schedule_interview).',
                    expected_output=f'Confirmation string that the email has been sent to {candidate_email}.',
                    agent=recruitment_coordinator,
                    context=[task_schedule_interview] # Depends on the scheduling task
                    # Arguments for email_tool will be determined by the LLM based on this description
                    # e.g. to_email, subject, body (which should include the Zoom link from context)
                )

                # Define Crew for coordination and scheduling
                recruitment_crew = Crew(
                    agents=[interview_scheduler, recruitment_coordinator], # Order can matter for sequential
                    tasks=[task_schedule_interview, task_send_interview_email],
                    process=Process.sequential,
                    verbose=True
                )

                st.info("🤝 Recruitment Crew is coordinating and scheduling...")
                try:
                    crew_result = recruitment_crew.kickoff()
                    st.subheader("Recruitment Process Outcome:")
                    st.markdown(crew_result) # Displays the output of the last task (email confirmation)

                    # Display individual task outputs for clarity
                    st.markdown("---")
                    st.markdown(f"**Zoom Scheduling Output:**\n {task_schedule_interview.output.exported_output if task_schedule_interview.output else 'Not available'}")
                    st.markdown(f"**Email Sending Output:**\n {task_send_interview_email.output.exported_output if task_send_interview_email.output else 'Not available'}")

                except Exception as e:
                    st.error(f"Error during recruitment crew execution: {e}")
                    import traceback
                    st.text(traceback.format_exc())
    elif st.session_state.screening_result: # Screening done, but not selected or email missing
        st.warning("Candidate not marked as 'Selected for Interview' based on screening, or candidate email is missing. Cannot proceed with application.")


st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, and various communication APIs.")

# Ensure all typing imports are at the top
from typing import Optional, List, Dict, Any
