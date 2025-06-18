import streamlit as st
import os
import json # Moved import json to the top
from typing import Optional # Added Optional for type hints
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, SerpDevTool

# Composio (phidata integration)
from composio.phidata.action import Action
from composio.phidata.toolset import ComposioToolSet
from composio.client.collections import App

# --- Load Environment Variables ---
load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY"
    # Prioritize environment variables
    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    # Fallback to session state (lowercase key for streamlit inputs)
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- Custom CrewAI Tool for Google Docs via Composio ---
class CreateGoogleDocTool(BaseTool):
    name: str = "Create Google Document Tool"
    description: str = "Creates a new Google Document with the given name and content (markdown)."
    composio_api_key: str

    def __init__(self, composio_api_key: str, **kwargs):
        super().__init__(**kwargs)
        if not composio_api_key:
            raise ValueError("Composio API key is required for CreateGoogleDocTool.")
        self.composio_api_key = composio_api_key
        # Initialize ComposioToolSet here or in _run.
        # Initializing here might be slightly more efficient if tool is used multiple times by an agent.
        self.toolset = ComposioToolSet(api_key=self.composio_api_key)
        # Enable the Google Docs tool specifically
        try:
            self.toolset.enable_app(App.GOOGLEDOCS)
        except Exception as e:
            # Allow initialization even if enabling fails here, _run will handle it.
            print(f"Warning: Could not pre-enable GoogleDocs in ComposioToolSet during init: {e}")


    def _run(self, document_name: str, content_markdown: str) -> str:
        try:
            # Ensure toolset is available
            if not self.toolset:
                 self.toolset = ComposioToolSet(api_key=self.composio_api_key)
                 self.toolset.enable_app(App.GOOGLEDOCS)

            gdocs_tool = self.toolset.get_action(Action.GOOGLEDOCS_CREATE_DOCUMENT)

            # Execute the Composio action.
            # The exact parameters depend on how Composio's GOOGLEDOCS_CREATE_DOCUMENT is defined.
            # Assuming it takes 'name' for document title and 'content' for the body.
            # Composio might expect content in a specific format (e.g., HTML, plain text, or markdown if it handles conversion).
            # For this example, let's assume it takes markdown and handles it.
            # If it expects HTML, markdown to HTML conversion would be needed here.

            # Placeholder for content if markdown is directly supported or if it's plain text
            # If Composio expects HTML, you'd convert content_markdown to HTML here.
            # For now, we pass markdown directly.

            # A common pattern for Composio tools is to pass parameters as a dictionary.
            # Let's assume the parameters are 'name' and 'document_content' or similar.
            # This needs to be verified against Composio's specific action schema.
            # For now, let's try with common sense names.
            # The Phidata example used `request_body={"name": name, "content": content}`
            # Let's try to match that structure if possible, or a simpler one.

            # The Phidata `execute` method for an action often takes `request_body`
            # Let's assume Composio's `execute` for this action takes named arguments or a dict.
            # If it's `execute(name=document_name, content=content_markdown)`:
            # response = gdocs_tool.execute(name=document_name, content=content_markdown)

            # If it's `execute(request_body={"name": "doc_name", "content": "doc_content"})`
            # This is more likely for complex tools.
            # The `composio-phidata` example for creating a Google Doc shows:
            # `gdocs_tool.execute(request_body={"name": "My Document Title", "content": "Hello World"})`
            # And the content was simple text. If we pass markdown, Google Docs should render it.

            response = gdocs_tool.execute(params={"name": document_name, "content": content_markdown})

            # Process the response to get the document link.
            # This depends on the structure of the response from Composio.
            # It might be in response['document_url'], response['url'], response['link'], etc.
            # Or it might be part of a more complex object.
            # Let's assume a common key like 'webViewLink' or 'document_url'.
            doc_link = response.get("document_url", response.get("webViewLink", response.get("url")))

            if doc_link:
                return f"Success: Google Doc '{document_name}' created. Link: {doc_link}"
            else:
                # Try to find an ID if no direct link
                doc_id = response.get("documentId", response.get("id"))
                if doc_id:
                    # Construct a typical Google Docs URL
                    doc_link_constructed = f"https://docs.google.com/document/d/{doc_id}/edit"
                    return f"Success: Google Doc '{document_name}' created. Link (constructed): {doc_link_constructed}. Raw response: {json.dumps(response)}"
                return f"Warning: Google Doc '{document_name}' created, but link not found in response. Response: {json.dumps(response)}"

        except Exception as e:
            return f"Error using CreateGoogleDocTool: {e}. Ensure Composio Google Docs app is connected and permissions are correct."

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Teaching Agent Team (CrewAI)")
st.title("📚 AI Teaching Agent Team (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state))
    st.session_state.composio_api_key = st.text_input("Composio API Key", type="password", value=get_api_key("COMPOSIO", st.session_state))
    st.session_state.serpapi_api_key = st.text_input("SerpAPI API Key", type="password", value=get_api_key("SERPAPI", st.session_state))

st.header("Enter the Topic for the Lesson Plan")
topic = st.text_input("Teaching Topic:", placeholder="e.g., Introduction to Photosynthesis")

if st.button("Generate Lesson Plan with CrewAI"):
    openai_api_key_val = get_api_key("OPENAI", st.session_state)
    composio_api_key_val = get_api_key("COMPOSIO", st.session_state)
    serpapi_api_key_val = get_api_key("SERPAPI", st.session_state)

    if not all([openai_api_key_val, composio_api_key_val, serpapi_api_key_val]):
        st.error("Please provide all API Keys: OpenAI, Composio, and SerpAPI.")
    elif not topic:
        st.error("Please enter a teaching topic.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key_val
        os.environ["SERP_API_KEY"] = serpapi_api_key_val # For SerpDevTool
        # Composio key is passed directly to its tool

        # Instantiate Tools
        try:
            google_doc_tool = CreateGoogleDocTool(composio_api_key=composio_api_key_val)
            search_tool = SerpDevTool() # Requires SERP_API_KEY in env
        except Exception as e:
            st.error(f"Error initializing tools: {e}")
            st.stop()


        # --- Define CrewAI Agents ---
        professor = Agent(
            role='University Professor',
            goal=f'Develop a comprehensive lecture on "{topic}", including key concepts, examples, and a summary. Then, save this lecture as a Google Document.',
            backstory='A seasoned professor with expertise in creating engaging and informative educational content across various subjects.',
            tools=[google_doc_tool],
            verbose=True,
            allow_delegation=False
        )
        academic_advisor = Agent(
            role='Academic Advisor',
            goal=f'Create a student study guide for the lecture on "{topic}", based on the professor\'s material. This guide should include learning objectives, key vocabulary, and review questions. Save this guide as a new Google Document.',
            backstory='An experienced academic advisor skilled in translating complex topics into student-friendly study materials.',
            tools=[google_doc_tool],
            verbose=True,
            allow_delegation=False
        )
        research_librarian = Agent(
            role='Research Librarian',
            goal=f'Compile a list of 5-7 external academic resources and references (articles, books, reputable websites) relevant to "{topic}". Save this bibliography as a Google Document.',
            backstory='A knowledgeable librarian with access to vast academic databases, adept at finding pertinent research materials and using web search tools.',
            tools=[google_doc_tool, search_tool],
            verbose=True,
            allow_delegation=False
        )
        teaching_assistant = Agent(
            role='Teaching Assistant (TA)',
            goal=f'Develop a short 5-question quiz with an answer key for the topic "{topic}", based on the professor\'s lecture and study guide. Also, find one relevant YouTube video link for further student engagement. Save the quiz (and video link) as a Google Document.',
            backstory='A helpful TA who supports student learning by creating assessments and finding supplementary materials, proficient with web search.',
            tools=[google_doc_tool, search_tool],
            verbose=True,
            allow_delegation=False
        )

        # --- Define CrewAI Tasks ---
        # Task descriptions instruct agents to use the tool and include the link.
        task_lecture = Task(
            description=f'Create a detailed lecture on "{topic}". The lecture should cover: 1. Introduction/Overview, 2. Core Concepts (at least 3-4), 3. Illustrative Examples for each concept, 4. A concise Summary. After generating the content, use the CreateGoogleDocTool to save it as "Lecture Notes: {topic}". Ensure the output of this task includes the phrase "Google Doc Link:" followed by the actual link.',
            expected_output=f'A confirmation message including the link to the Google Document containing the lecture notes for "{topic}". For example: "Lecture created. Google Doc Link: http://docs.google.com/..."',
            agent=professor
        )
        task_study_guide = Task(
            description=f'Based on the lecture content for "{topic}" (provided as context), create a student study guide. The guide must include: 1. Clear Learning Objectives (3-5 points), 2. Key Vocabulary (5-7 terms with definitions), 3. Review Questions (5 open-ended questions). After generating the content, use the CreateGoogleDocTool to save it as "Study Guide: {topic}". Ensure the output includes "Google Doc Link:" followed by the link.',
            expected_output=f'A confirmation message including the link to the Google Document containing the study guide for "{topic}".',
            agent=academic_advisor,
            context=[task_lecture] # Depends on the professor's lecture
        )
        task_bibliography = Task(
            description=f'Compile a bibliography of 5-7 relevant external academic resources for the topic "{topic}". Include journal articles, book chapters, or highly reputable educational websites. For each resource, provide a brief annotation (1-2 sentences). Use the search tool for finding resources. After compiling, use the CreateGoogleDocTool to save it as "Bibliography: {topic}". Ensure the output includes "Google Doc Link:" followed by the link.',
            expected_output=f'A confirmation message including the link to the Google Document containing the bibliography for "{topic}".',
            agent=research_librarian
        )
        task_quiz = Task(
            description=f'Develop a 5-question quiz (multiple choice or short answer) with a corresponding answer key for "{topic}", based on the lecture and study guide (provided as context). Also, find one relevant and high-quality YouTube video link related to "{topic}" for students. After creating the quiz and finding the video, use the CreateGoogleDocTool to save the quiz, answer key, and video link as "Quiz and Resources: {topic}". Ensure the output includes "Google Doc Link:" followed by the link.',
            expected_output=f'A confirmation message including the link to the Google Document containing the quiz, answer key, and YouTube video link for "{topic}".',
            agent=teaching_assistant,
            context=[task_lecture, task_study_guide] # Depends on lecture and study guide
        )

        # --- Define Crew ---
        teaching_crew = Crew(
            agents=[professor, academic_advisor, research_librarian, teaching_assistant],
            tasks=[task_lecture, task_study_guide, task_bibliography, task_quiz],
            process=Process.sequential,
            verbose=True
        )

        st.info("🧑‍🏫 AI Teaching Team is generating the lesson plan... This may take a few minutes per document.")

        crew_inputs = {'topic': topic} # This is implicitly used by f-strings in descriptions

        try:
            crew_result_summary = teaching_crew.kickoff(inputs=crew_inputs)

            st.subheader("🎓 Lesson Plan Generation Complete!")
            st.markdown("---")

            st.markdown("### Overall Crew Summary/Output (Last Task):")
            st.write(crew_result_summary) # Output of the last task (TA's quiz)
            st.markdown("---")

            st.markdown("### Individual Agent Outputs & Google Doc Links:")
            for i, task_item in enumerate(teaching_crew.tasks):
                agent_name = teaching_crew.agents[i].role # Assuming one agent per task in sequential setup
                st.markdown(f"**Output from {agent_name} (Task: {task_item.description[:50]}...):**")
                if task_item.output:
                    st.markdown(task_item.output.exported_output)
                    # Attempt to find a Google Doc link in the output
                    output_text = str(task_item.output.exported_output)
                    link_start = output_text.find("http://docs.google.com/")
                    if link_start == -1:
                         link_start = output_text.find("https://docs.google.com/")

                    if link_start != -1:
                        link_end = output_text.find(" ", link_start)
                        link_end_newline = output_text.find("\n", link_start)
                        if link_end == -1 or (link_end_newline != -1 and link_end_newline < link_end) : link_end = link_end_newline

                        doc_link_found = output_text[link_start:] if link_end == -1 else output_text[link_start:link_end]
                        st.markdown(f"🔗 [Open {agent_name}'s Google Doc]({doc_link_found.strip()})")
                    else:
                        st.warning(f"Could not automatically find a Google Doc link in the output for {agent_name}.")

                else:
                    st.warning(f"No output recorded for {agent_name}'s task.")
                st.markdown("---")

        except Exception as e:
            st.error(f"Error during CrewAI execution: {e}")
            import traceback
            st.text(traceback.format_exc())

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, Composio (Google Docs), and SerpAPI.")
