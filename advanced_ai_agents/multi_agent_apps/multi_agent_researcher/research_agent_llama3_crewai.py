import streamlit as st
import os
from dotenv import load_dotenv

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.llms import Ollama # For Llama3 local models
from crewai_tools import HackerNewsSearchTool

load_dotenv()

# --- Helper for Configuration ---
def get_config_value(config_key: str, session_state, default_value: str = ""):
    # For environment variables, keys are usually uppercase
    env_var_name = config_key.upper()
    if env_var_name in os.environ:
        return os.environ[env_var_name]
    # For Streamlit inputs, keys are usually lowercase in session_state
    return session_state.get(config_key.lower(), default_value)

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Hacker News Researcher (Llama3 CrewAI)")
st.title("📰 Hacker News Research Agent Team (Llama3 & CrewAI)")

with st.sidebar:
    st.header("Ollama Configuration")
    st.session_state.ollama_base_url = st.text_input(
        "Ollama Server URL",
        value=get_config_value("OLLAMA_BASE_URL", st.session_state, "http://localhost:11434")
    )
    st.session_state.ollama_model_name = st.text_input(
        "Ollama Model Name",
        value=get_config_value("OLLAMA_MODEL_NAME", st.session_state, "llama3") # Default to llama3
    )
    st.info(
        "Ensure your Ollama server is running and the specified model (e.g., 'llama3', 'llama3.1', 'mixtral') is downloaded. "
        "This app uses the HackerNewsSearchTool from crewai-tools."
    )

st.header("🔬 Enter Your Research Query")
research_query = st.text_input(
    "What topic on Hacker News are you interested in?",
    placeholder="e.g., 'AI advancements in healthcare', 'new programming languages gaining traction'"
)

if st.button("🚀 Start Research with Llama3", type="primary"):
    ollama_base_url_val = get_config_value("ollama_base_url", st.session_state, "http://localhost:11434")
    ollama_model_name_val = get_config_value("ollama_model_name", st.session_state, "llama3")

    if not ollama_base_url_val or not ollama_model_name_val:
        st.error("❌ Ollama Server URL and Model Name are required. Please configure them in the sidebar.")
    elif not research_query:
        st.error("❌ Please enter a research query.")
    else:
        # Initialize Tools
        try:
            hn_search_tool = HackerNewsSearchTool()
        except Exception as e:
            st.error(f"Error initializing HackerNewsSearchTool: {e}. Ensure crewai-tools are correctly installed.")
            st.stop()

        # Define LLM using Ollama
        try:
            # Ensure Ollama LLM is configured correctly for CrewAI
            # The 'Ollama' class from crewai.llms typically takes `model` and `base_url`.
            llm = Ollama(model=ollama_model_name_val, base_url=ollama_base_url_val)
            # Perform a quick test if possible (e.g. generate a short text)
            # This is tricky as it's an async call usually, and CrewAI handles it.
            # For now, assume it will work if parameters are correct.
            st.success(f"Ollama LLM initialized with model: {ollama_model_name_val} at {ollama_base_url_val}")
        except Exception as e:
            st.error(f"Error initializing Ollama LLM: {e}. Check your Ollama server and model name.")
            st.stop()


        # --- Define CrewAI Agents (using Ollama LLM) ---
        story_scout = Agent(
            role='Hacker News Story Scout (Llama3)',
            goal=f'Find the top 5-7 most relevant and recent Hacker News stories related to the query: "{research_query}".',
            backstory=(
                "An expert in navigating Hacker News to quickly find trending and significant stories using local LLM capabilities. "
                "You are skilled at using search tools to filter out noise and identify key discussions."
            ),
            llm=llm,
            tools=[hn_search_tool],
            verbose=True,
            allow_delegation=False
        )

        content_processor = Agent(
            role='Hacker News Content Processor (Llama3)',
            goal=(
                'For each identified Hacker News story, extract the main topic, a brief summary of the article/discussion, '
                'and identify any particularly insightful comments or viewpoints if available from the search tool output. '
                'Focus on the information provided by the HackerNewsSearchTool.'
            ),
            backstory=(
                "A meticulous analyst who excels at distilling key information from Hacker News story summaries and comments, powered by a local LLM. "
                "You understand that the HackerNewsSearchTool provides summaries and sometimes snippets of content, "
                "and your job is to process this available information effectively."
            ),
            llm=llm,
            tools=[hn_search_tool],
            verbose=True,
            allow_delegation=False
        )

        chief_synthesizer = Agent(
            role='Chief HN Synthesis Agent (Llama3)',
            goal=(
                'Synthesize the processed information from multiple Hacker News stories into a coherent and insightful report. '
                'The report should summarize the key findings, trends, and overall sentiment regarding "{research_query}".'
            ),
            backstory=(
                "A strategic thinker and excellent communicator with a talent for seeing the bigger picture, using local LLM power. "
                "You can connect disparate pieces of information from various sources to create a comprehensive overview."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        # --- Define CrewAI Tasks (same structure as OpenAI version) ---
        find_stories_task = Task(
            description=f'Search Hacker News for top and recent stories related to: "{research_query}". Identify 5-7 relevant stories. Your output should be a list of these stories, including their titles, URLs, and any available summaries or scores from the search tool.',
            expected_output=(
                "A list of dictionaries, where each dictionary represents a Hacker News story and contains at least 'title', 'link', "
                "and 'summary' (if available from the tool's output). For example: "
                "[{'title': 'Story 1', 'link': 'http://example.com/story1', 'summary': 'Summary of story 1...'}, ...]"
            ),
            agent=story_scout
        )

        process_stories_task = Task(
            description=(
                "For each story provided in the context (from the Story Scout), analyze the available information (title, summary, etc.). "
                "Extract the main topic of each story and a concise summary of its content or discussion. "
                "If the search tool provided snippets of comments, note any highly insightful ones. "
                "Your goal is to prepare structured data for each story for the synthesizer."
            ),
            expected_output=(
                "A list of processed story objects. Each object should contain: 'original_title', 'original_link', "
                "'processed_topic', 'content_summary', and 'insightful_comment_snippets' (if any). For example: "
                "[{'original_title': 'Story 1', 'original_link': '...', 'processed_topic': 'AI in X', "
                "'content_summary': 'This story discusses...', 'insightful_comment_snippets': ['Comment A...']}, ...]"
            ),
            agent=content_processor,
            context=[find_stories_task]
        )

        synthesize_report_task = Task(
            description=(
                f'Take the processed information about multiple Hacker News stories related to "{research_query}" (from the Content Processor). '
                "Synthesize this information into a single, coherent report. The report should cover:\n"
                "1. An overall summary of the findings regarding the research query.\n"
                "2. Key themes or topics that emerged across the stories.\n"
                "3. Any notable trends, technologies, or opinions discussed.\n"
                "4. A concluding thought on the significance of these discussions for the topic: '{research_query}'.\n"
                "The report should be well-structured and insightful, suitable for a technical audience."
            ),
            expected_output=(
                "A comprehensive markdown report summarizing the research findings from Hacker News on '{research_query}'. "
                "The report should be at least 3-4 paragraphs long and cover the points mentioned in the description."
            ),
            agent=chief_synthesizer,
            context=[process_stories_task]
        )

        # Define Crew
        research_crew = Crew(
            agents=[story_scout, content_processor, chief_synthesizer],
            tasks=[find_stories_task, process_stories_task, synthesize_report_task],
            process=Process.sequential,
            verbose=True # Set to True for more detailed logs in terminal during development
        )

        st.info(f"🚀 Kicking off Hacker News research for: \"{research_query}\" using Llama3...")

        with st.spinner(f"🕵️ Llama3-powered agents are researching Hacker News... This may take some time depending on your local Ollama setup..."):
            try:
                crew_result = research_crew.kickoff(inputs={'research_query': research_query})

                st.subheader("📈 Research Report (Llama3):")
                st.markdown("---")
                st.markdown(crew_result)
                st.success("Research complete!")

            except Exception as e:
                st.error(f"Error during Llama3 CrewAI execution: {e}")
                st.error("Common issues: \n"
                         "- Ollama server not running or not accessible at the specified URL.\n"
                         "- The specified Ollama model is not downloaded/available (e.g., `ollama pull llama3`).\n"
                         "- Network issues or tool errors (HackerNewsSearchTool).")
                import traceback
                st.text(traceback.format_exc())

st.markdown("---")
st.caption("Powered by CrewAI (with local Llama3 via Ollama), Streamlit, and Hacker News Search.")
