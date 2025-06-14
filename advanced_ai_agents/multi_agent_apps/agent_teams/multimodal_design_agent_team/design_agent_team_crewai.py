import streamlit as st
import os
from PIL import Image
import io
import base64
# Typing import moved to top section
from dotenv import load_dotenv

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import DuckDuckGoSearchRun
from crewai.llms import ChatGoogleGenerativeAI

load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY" # e.g., GOOGLE_GEMINI_API_KEY
    # CrewAI's ChatGoogleGenerativeAI uses GOOGLE_API_KEY by default in its environment loading
    if service_name.upper() == "GOOGLE_GEMINI": # Special handling for Gemini key for CrewAI
        env_var_key = "GOOGLE_API_KEY"

    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    # Fallback to session state (lowercase key for streamlit inputs)
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- Image Processing ---
def process_uploaded_images(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Dict[str, Any]]:
    """Converts uploaded images to a list of dicts with name and base64 data."""
    processed_images = []
    if not uploaded_files:
        return processed_images

    for uploaded_file in uploaded_files:
        try:
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode()
            processed_images.append({
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "base64": base64_image
            })
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {e}")
    return processed_images

def format_image_for_gemini(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """Formats a single processed image for Gemini LLM input."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image_data['type']};base64,{image_data['base64']}"}
    }

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Multimodal Design Analysis Team (CrewAI)")
st.title("🎨 AI Multimodal Design Analysis Team (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    # Note: For CrewAI's ChatGoogleGenerativeAI, the key is often expected as GOOGLE_API_KEY
    st.session_state.google_gemini_api_key = st.text_input(
        "Google Gemini API Key (sets GOOGLE_API_KEY env var)",
        type="password",
        value=get_api_key("GOOGLE_GEMINI", st.session_state) # This helper will check GOOGLE_API_KEY
    )
    st.session_state.serpapi_api_key = st.text_input("SerpAPI API Key (for Market Research)", type="password", value=get_api_key("SERPAPI", st.session_state))

st.header("🖼️ Upload Designs")
primary_design_files = st.file_uploader("Upload Primary Design Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
competitor_design_files = st.file_uploader("Upload Competitor Design Image(s) (Optional, for Market Analysis)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.header("⚙️ Analysis Configuration")
analysis_types = st.multiselect(
    "Select Analysis Types to Perform:",
    options=["Visual Design", "User Experience", "Market Analysis"],
    default=["Visual Design", "User Experience"]
)
custom_focus_points = st.text_area("Specific Focus Points or Questions for the Analysis (Optional):", placeholder="e.g., 'Focus on the color palette and typography.' or 'How does the UX compare to Material Design principles?'")


if st.button("🚀 Run Design Analysis with CrewAI"):
    gemini_key = get_api_key("GOOGLE_GEMINI", st.session_state)
    serpapi_key = get_api_key("SERPAPI", st.session_state)

    if not gemini_key:
        st.error("Google Gemini API Key (GOOGLE_API_KEY) is required.")
    elif not primary_design_files:
        st.error("Please upload at least one Primary Design Image.")
    elif not analysis_types:
        st.error("Please select at least one Analysis Type.")
    else:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        if serpapi_key:
            os.environ["SERPAPI_API_KEY"] = serpapi_key

        processed_primary_images = process_uploaded_images(primary_design_files)
        processed_competitor_images = process_uploaded_images(competitor_design_files)

        if not processed_primary_images:
            st.error("Failed to process primary design images. Please try again.")
            st.stop()

        # Initialize LLM for CrewAI
        try:
            # Using gemini-1.5-flash-latest as gemini-pro-vision is legacy for some uses.
            # Adjust model name as per current Gemini availability and needs.
            gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_key)
        except Exception as e:
            st.error(f"Failed to initialize Gemini LLM: {e}. Ensure the API key is correct and the model name is valid.")
            st.stop()

        # Initialize Tools
        search_tool = DuckDuckGoSearchRun() if "Market Analysis" in analysis_types and not serpapi_key else SerpDevTool() if "Market Analysis" in analysis_types and serpapi_key else None
        if "Market Analysis" in analysis_types and not search_tool:
             st.warning("SerpAPI key not provided, Market Research agent will use DuckDuckGo (less effective).")
             search_tool = DuckDuckGoSearchRun()


        # --- Agent Definitions ---
        visual_analyst = Agent(
            role='Visual Design Analyst',
            goal='Analyze the visual aesthetics, branding, typography, color schemes, and overall visual appeal of the provided design images. Provide actionable feedback for improvement.',
            backstory='An expert UI/UX designer with a keen eye for visual details and current design trends. Specializes in providing constructive criticism on visual design elements.',
            llm=gemini_llm,
            verbose=True,
            allow_delegation=False
        )
        ux_analyst = Agent(
            role='User Experience (UX) Analyst',
            goal='Evaluate the user experience, information architecture, navigation, usability, and accessibility of the provided design images. Identify pain points and suggest UX enhancements.',
            backstory='A seasoned UX researcher and designer focused on creating intuitive and user-centered digital experiences. Expert in heuristic evaluation and usability testing principles.',
            llm=gemini_llm,
            verbose=True,
            allow_delegation=False
        )
        market_researcher = Agent(
            role='Market Research Analyst (Design Focus)',
            goal='Analyze the provided primary design images in the context of current market trends and competitor designs (if provided). Identify unique selling propositions, potential market fit, and areas for differentiation.',
            backstory='A strategic market analyst specializing in the design and tech industry. Uses web search and comparative analysis to provide insights on market positioning.',
            llm=gemini_llm,
            tools=[search_tool] if search_tool else [],
            verbose=True,
            allow_delegation=False
        )

        all_tasks = []
        results_display = {}

        # For passing image data to tasks/LLM.
        # Gemini expects content in a specific list format: [text_part, image_part_1, image_part_2, ...]
        # We'll construct this for each task that needs images.

        primary_image_inputs_for_llm = [format_image_for_gemini(img) for img in processed_primary_images]
        competitor_image_inputs_for_llm = [format_image_for_gemini(img) for img in processed_competitor_images]


        st.info(f"Starting analysis for: {', '.join(analysis_types)}")

        for analysis_type in analysis_types:
            task_description_parts = []
            current_agent = None
            task_context = None # No inter-task context for now, each is independent.

            if analysis_type == "Visual Design":
                current_agent = visual_analyst
                task_description_parts.append(f"Analyze the visual design of the primary image(s) named: {[img['name'] for img in processed_primary_images]}.")
                task_description_parts.append("Focus on: overall aesthetics, branding consistency, typography, color palette, imagery, layout, and visual hierarchy.")
                if custom_focus_points: task_description_parts.append(f"User's specific focus: {custom_focus_points}")
                task_description_parts.append("Provide a detailed report with observations and actionable recommendations for improvement.")

                # Directly invoke LLM for multimodal input as CrewAI task inputs are primarily text.
                # This approach is similar to the multimodal coding agent.
                content_for_llm = [{"type": "text", "text": "\n".join(task_description_parts)}] + primary_image_inputs_for_llm

            elif analysis_type == "User Experience":
                current_agent = ux_analyst
                task_description_parts.append(f"Analyze the user experience (UX) of the primary image(s) named: {[img['name'] for img in processed_primary_images]}.")
                task_description_parts.append("Focus on: usability, navigation flow, information architecture, clarity of calls to action, accessibility considerations, and potential user pain points.")
                if custom_focus_points: task_description_parts.append(f"User's specific focus: {custom_focus_points}")
                task_description_parts.append("Provide a detailed UX evaluation with actionable recommendations.")
                content_for_llm = [{"type": "text", "text": "\n".join(task_description_parts)}] + primary_image_inputs_for_llm

            elif analysis_type == "Market Analysis":
                current_agent = market_researcher
                task_description_parts.append(f"Conduct a market analysis for the primary design(s) named: {[img['name'] for img in processed_primary_images]}.")
                if processed_competitor_images:
                    task_description_parts.append(f"Compare it against competitor designs named: {[img['name'] for img in processed_competitor_images]}. The competitor images are also provided.")
                else:
                    task_description_parts.append("No specific competitor images provided; focus on general market trends and positioning.")
                task_description_parts.append("Use web search if needed for trends. Focus on: market positioning, differentiation opportunities, alignment with current design trends, and target audience appeal.")
                if custom_focus_points: task_description_parts.append(f"User's specific focus: {custom_focus_points}")
                task_description_parts.append("Provide a market research report with strategic insights.")

                # Construct content for LLM, including competitor images if available
                content_for_llm = [{"type": "text", "text": "\n".join(task_description_parts)}] + primary_image_inputs_for_llm
                if processed_competitor_images:
                    content_for_llm.append({"type": "text", "text": "\n\nCompetitor Images Analysis:"}) # Separator
                    content_for_llm.extend(competitor_image_inputs_for_llm)

            if current_agent:
                with st.spinner(f"🧠 {current_agent.role} is performing {analysis_type} analysis..."):
                    try:
                        # Direct LLM invocation for multimodal tasks
                        from langchain_core.messages import HumanMessage
                        response = current_agent.llm.invoke([HumanMessage(content=content_for_llm)])
                        analysis_result = response.content # Assuming response structure has .content
                        results_display[analysis_type] = analysis_result
                        st.success(f"{analysis_type} analysis complete by {current_agent.role}.")
                    except Exception as e:
                        st.error(f"Error during {analysis_type} analysis by {current_agent.role}: {e}")
                        results_display[analysis_type] = f"Error: {e}"
                        import traceback
                        st.text(traceback.format_exc())

        st.markdown("---")
        st.header("📊 Combined Analysis Results:")
        for analysis_type, result_text in results_display.items():
            with st.expander(f"{analysis_type} Report", expanded=True):
                st.markdown(result_text)

st.markdown("---")
st.caption("Powered by CrewAI (with direct Gemini calls for multimodal), Streamlit.")

# Ensure all typing imports are at the top
from typing import List, Dict, Any, Optional
