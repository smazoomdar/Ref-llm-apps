import streamlit as st
import os
from dotenv import load_dotenv

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.llms import ChatOpenAI
from crewai_tools import HackerNewsSearchTool

load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY"
    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Hacker News Researcher (CrewAI)")
st.title("📰 Hacker News Research Agent Team (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=get_api_key("OPENAI", st.session_state)
    )
    st.info("This app uses the HackerNewsSearchTool from crewai-tools to find relevant stories based on your query.")

st.header("🔬 Enter Your Research Query")
research_query = st.text_input(
    "What topic on Hacker News are you interested in?",
    placeholder="e.g., 'AI advancements in healthcare', 'new programming languages gaining traction'"
)

if st.button("🚀 Start Research", type="primary"):
    openai_key = get_api_key("OPENAI", st.session_state)

    if not openai_key:
        st.error("❌ OpenAI API Key is required. Please enter it in the sidebar.")
    elif not research_query:
        st.error("❌ Please enter a research query.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key

        # Initialize Tools
        try:
            hn_search_tool = HackerNewsSearchTool()
            # Test the tool with a generic query to ensure it's working (optional, but good for debugging)
            # try:
            #     print(f"Testing HackerNewsSearchTool with query: 'ai'")
            #     test_results = hn_search_tool.run(search_query='ai') # The tool's run method might have a different signature
            #     print(f"HackerNewsSearchTool test results: {test_results[:50]}...") # Print snippet
            # except Exception as e:
            #     st.warning(f"HackerNewsSearchTool might not be fully working: {e}. Proceeding anyway.")
        except Exception as e:
            st.error(f"Error initializing HackerNewsSearchTool: {e}. Ensure crewai-tools are correctly installed.")
            st.stop()

        # Define LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7) # or gpt-3.5-turbo

        # --- Define CrewAI Agents ---
        story_scout = Agent(
            role='Hacker News Story Scout',
            goal=f'Find the top 5-7 most relevant and recent Hacker News stories related to the query: "{research_query}".',
            backstory=(
                "An expert in navigating Hacker News to quickly find trending and significant stories. "
                "You are skilled at using search tools to filter out noise and identify key discussions."
            ),
            llm=llm,
            tools=[hn_search_tool],
            verbose=True,
            allow_delegation=False
        )

        content_processor = Agent(
            role='Hacker News Content Processor',
            goal=(
                'For each identified Hacker News story, extract the main topic, a brief summary of the article/discussion, '
                'and identify any particularly insightful comments or viewpoints if available from the search tool output. '
                'Focus on the information provided by the HackerNewsSearchTool.'
            ),
            backstory=(
                "A meticulous analyst who excels at distilling key information from Hacker News story summaries and comments. "
                "You understand that the HackerNewsSearchTool provides summaries and sometimes snippets of content, "
                "and your job is to process this available information effectively."
            ),
            llm=llm,
            tools=[hn_search_tool], # May use it to re-fetch or get more details if the tool supports it for specific story IDs.
                                 # The current HN tool in crewai_tools might be search-only.
                                 # If so, this agent will primarily work with the summaries from the scout.
            verbose=True,
            allow_delegation=False
        )

        chief_synthesizer = Agent(
            role='Chief HN Synthesis Agent',
            goal=(
                'Synthesize the processed information from multiple Hacker News stories into a coherent and insightful report. '
                'The report should summarize the key findings, trends, and overall sentiment regarding "{research_query}".'
            ),
            backstory=(
                "A strategic thinker and excellent communicator with a talent for seeing the bigger picture. "
                "You can connect disparate pieces of information from various sources to create a comprehensive overview."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False # No tools needed for pure synthesis from context
        )

        # --- Define CrewAI Tasks ---
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
                "The report should be well-structured and insightful."
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
            verbose=True
        )

        st.info(f"🚀 Kicking off Hacker News research for: \"{research_query}\"...")

        with st.spinner("🕵️ Agents are researching Hacker News... This may take a few moments..."):
            try:
                crew_result = research_crew.kickoff(inputs={'research_query': research_query})

                st.subheader("📈 Research Report:")
                st.markdown("---")
                st.markdown(crew_result) # Display final report from the synthesizer
                st.success("Research complete!")

            except Exception as e:
                st.error(f"Error during CrewAI execution: {e}")
                import traceback
                st.text(traceback.format_exc())

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, and Hacker News Search.")
