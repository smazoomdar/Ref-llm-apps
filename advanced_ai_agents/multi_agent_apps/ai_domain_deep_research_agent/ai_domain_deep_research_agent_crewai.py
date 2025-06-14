import streamlit as st
import os
import json
import re # Moved import re to the top
from typing import List, Dict, Any, Optional # Ensured typing imports are here
from dotenv import load_dotenv

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, DuckDuckGoSearchRun, SerpDevTool
from crewai.llms import ChatOpenAI # Placeholder if TogetherAI is complex to integrate directly initially
# from langchain_together import ChatTogether # For potential TogetherAI integration

# Composio (phidata integration)
from composio.phidata.action import Action
from composio.phidata.toolset import ComposioToolSet
from composio.client.collections import App


load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY"
    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- LLM Configuration ---
def get_llm(openai_api_key: Optional[str] = None, together_api_key: Optional[str] = None):
    # Prioritize TogetherAI if key is provided, otherwise fallback to OpenAI or error
    if together_api_key:
        st.info("Attempting to use TogetherAI LLM (conceptual). Ensure ChatTogether or CrewAI's TogetherLLM is correctly set up.")
        # This is where you'd integrate ChatTogether with CrewAI, possibly via CustomLLM
        # For now, this is a placeholder. If direct integration is not straightforward,
        # we might need a custom wrapper or rely on OpenAI as a fallback.
        # Example (conceptual, assuming ChatTogether works like other Langchain LLMs for CrewAI):
        # try:
        #     from langchain_together import ChatTogether
        #     return ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", together_api_key=together_api_key, temperature=0.7)
        # except ImportError:
        #     st.warning("langchain-together not installed. Falling back to OpenAI if available.")
        # For this exercise, as direct ChatTogether with CrewAI's default LLM registration might be tricky,
        # we'll use OpenAI and note this as an area for specific TogetherAI integration.
        st.warning("Using OpenAI as a placeholder for TogetherAI LLM in this version.")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            return ChatOpenAI(model_name="gpt-4o", temperature=0.7) # Or "gpt-3.5-turbo"
        else:
            st.error("TogetherAI selected but OpenAI key (as fallback) also missing.")
            return None
    elif openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    else:
        st.error("No LLM API Key provided (OpenAI or TogetherAI).")
        return None


# --- Custom Tools ---
class ComposioBasedGoogleDocTool(BaseTool):
    name: str = "Google Document Creator (via Composio)"
    description: str = "Creates a new Google Document with the given name and markdown content using Composio."
    composio_api_key: str
    toolset: Optional[ComposioToolSet] = None

    def __init__(self, composio_api_key: str, **kwargs):
        super().__init__(**kwargs)
        if not composio_api_key:
            raise ValueError("Composio API key is required for ComposioBasedGoogleDocTool.")
        self.composio_api_key = composio_api_key
        try:
            self.toolset = ComposioToolSet(api_key=self.composio_api_key)
            self.toolset.enable_app(App.GOOGLEDOCS)
        except Exception as e:
            self.toolset = None # Ensure toolset is None if init fails
            # Deferring error to _run allows app to load if Composio is temporarily down
            print(f"Warning: ComposioBasedGoogleDocTool failed to initialize ComposioToolSet or enable GoogleDocs during __init__: {e}")


    def _run(self, document_name: str, content_markdown: str) -> str:
        if not self.toolset:
            # Attempt to re-initialize if failed during __init__
            try:
                self.toolset = ComposioToolSet(api_key=self.composio_api_key)
                self.toolset.enable_app(App.GOOGLEDOCS)
            except Exception as e:
                 return f"Error: ComposioToolSet could not be initialized: {e}"

        try:
            gdocs_tool_action = self.toolset.get_action(Action.GOOGLEDOCS_CREATE_DOCUMENT)
            # Based on Phidata examples, content might need to be plain text for their default create action.
            # If markdown is passed, Google Docs will likely render it as plain text with markdown syntax.
            # For true markdown rendering, content_markdown would need conversion to HTML or Google Docs API specific format.
            # For simplicity, passing markdown as "content".
            response = gdocs_tool_action.execute(params={"name": document_name, "content": content_markdown})

            doc_link = response.get("document_url", response.get("webViewLink"))
            doc_id = response.get("documentId", response.get("id"))

            if doc_link:
                return f"Success: Google Doc '{document_name}' created. Link: {doc_link}"
            elif doc_id:
                constructed_link = f"https://docs.google.com/document/d/{doc_id}/edit"
                return f"Success: Google Doc '{document_name}' created. Link (constructed): {constructed_link}. Raw response: {json.dumps(response)}"
            else:
                return f"Warning: Google Doc '{document_name}' created, but link not found in response. Response: {json.dumps(response)}"
        except Exception as e:
            return f"Error using ComposioBasedGoogleDocTool: {e}. Ensure Composio Google Docs app is connected and permissions are correct."


# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Domain Deep Research Agent (CrewAI)")
st.title("🔬 AI Domain Deep Research Agent (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    # Using OpenAI as the primary/fallback LLM for this example due to potential complexities with TogetherAI direct integration in CrewAI
    st.session_state.openai_api_key = st.text_input("OpenAI API Key (Used if TogetherAI not configured/fails)", type="password", value=get_api_key("OPENAI", st.session_state))
    st.session_state.togetherai_api_key = st.text_input("TogetherAI API Key (Optional)", type="password", value=get_api_key("TOGETHERAI", st.session_state))
    st.session_state.composio_api_key = st.text_input("Composio API Key (for Google Docs)", type="password", value=get_api_key("COMPOSIO", st.session_state))
    st.session_state.serpapi_api_key = st.text_input("SerpAPI API Key (for Search, Optional)", type="password", value=get_api_key("SERPAPI", st.session_state))

st.header("Research Parameters")
topic = st.text_input("Research Topic:", placeholder="e.g., The Impact of Quantum Computing on Cybersecurity")
domain = st.text_input("Specific Domain/Area of Focus:", placeholder="e.g., Financial Services, Healthcare AI, etc.")

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if 'final_report_link' not in st.session_state:
    st.session_state.final_report_link = None


# --- Step 1: Generate Research Questions ---
if st.button("1. Generate Research Questions"):
    openai_key_val = get_api_key("OPENAI", st.session_state)
    together_key_val = get_api_key("TOGETHERAI", st.session_state)

    if not topic or not domain:
        st.error("Please provide both Research Topic and Domain.")
    elif not (openai_key_val or together_key_val):
        st.error("Please provide an LLM API Key (OpenAI or TogetherAI).")
    else:
        llm_instance = get_llm(openai_api_key=openai_key_val, together_api_key=together_key_val)
        if not llm_instance:
            st.stop()

        question_generator = Agent(
            role='Expert Question Generator',
            goal=f'Generate a list of 5-7 insightful and comprehensive research questions about "{topic}" within the domain of "{domain}". These questions should guide a deep dive into the topic.',
            backstory='A seasoned academic and researcher skilled at formulating pivotal questions that drive thorough investigation.',
            llm=llm_instance,
            verbose=True,
            allow_delegation=False
        )
        question_task = Task(
            description=f'Generate 5 to 7 key research questions for the topic: "{topic}" focused on the "{domain}" domain. The questions should be probing and cover various aspects like challenges, opportunities, current state, future trends, etc. Return the questions as a numbered list.',
            expected_output='A numbered list of 5-7 research questions. Each question should be on a new line. Example:\n1. What are the primary challenges...\n2. How does X impact Y...',
            agent=question_generator
        )
        temp_crew = Crew(agents=[question_generator], tasks=[question_task], process=Process.sequential, verbose=0)

        with st.spinner("Generating research questions..."):
            try:
                result = temp_crew.kickoff(inputs={'topic': topic, 'domain': domain})
                # Assuming result is a string with numbered questions
                st.session_state.questions = [q.strip() for q in result.strip().split('\n') if q.strip() and q[0].isdigit()]
                st.success("Research questions generated!")
            except Exception as e:
                st.error(f"Error generating questions: {e}")
                st.session_state.questions = []

if st.session_state.questions:
    st.subheader("Generated Research Questions:")
    for i, q_text in enumerate(st.session_state.questions):
        st.markdown(f"{q_text}") # Questions are already numbered from agent

# --- Step 2: Start Research ---
if st.session_state.questions and st.button("2. Start Research on Generated Questions"):
    openai_key_val = get_api_key("OPENAI", st.session_state)
    together_key_val = get_api_key("TOGETHERAI", st.session_state)
    serpapi_key_val = get_api_key("SERPAPI", st.session_state)

    if not (openai_key_val or together_key_val):
        st.error("LLM API Key needed for research.")
    else:
        llm_instance = get_llm(openai_api_key=openai_key_val, together_api_key=together_key_val)
        if not llm_instance:
            st.stop()

        search_tool_instance = None
        if serpapi_key_val:
            os.environ["SERPAPI_API_KEY"] = serpapi_key_val
            search_tool_instance = SerpDevTool()
        else:
            st.warning("SerpAPI key not provided. Using DuckDuckGo for web search (may be less effective).")
            search_tool_instance = DuckDuckGoSearchRun()

        research_agent = Agent(
            role='Dedicated Researcher',
            goal='Thoroughly research a given question using available tools (web search) and provide a comprehensive answer.',
            backstory='A meticulous researcher who leaves no stone unturned to find accurate and detailed information.',
            llm=llm_instance,
            tools=[search_tool_instance] if search_tool_instance else [],
            verbose=True,
            allow_delegation=False
        )

        st.session_state.qa_pairs = [] # Reset previous research
        research_progress_bar = st.progress(0)
        total_questions = len(st.session_state.questions)

        for i, question_text in enumerate(st.session_state.questions):
            # Extract the actual question if it's numbered like "1. Question text"
            actual_question = re.sub(r"^\d+\.\s*", "", question_text)

            st.markdown(f"**Researching Question {i+1}/{total_questions}:** _{actual_question}_")
            research_task = Task(
                description=f'Research and answer the following question related to "{topic}" in the "{domain}" domain: "{actual_question}". Provide a detailed and well-supported answer.',
                expected_output='A comprehensive answer to the question, citing sources if web search was used. The answer should be a few paragraphs long.',
                agent=research_agent
            )
            temp_crew = Crew(agents=[research_agent], tasks=[research_task], process=Process.sequential, verbose=0)

            with st.spinner(f"Researching: {actual_question[:50]}..."):
                try:
                    answer = temp_crew.kickoff(inputs={'question': actual_question})
                    st.session_state.qa_pairs.append({"question": actual_question, "answer": answer})
                    st.info(f"Answer for Q{i+1} received.")
                except Exception as e:
                    st.error(f"Error researching question '{actual_question}': {e}")
                    st.session_state.qa_pairs.append({"question": actual_question, "answer": f"Error: Could not research this question. {e}"})
            research_progress_bar.progress((i + 1) / total_questions)
        st.success("Research phase complete!")


if st.session_state.qa_pairs:
    st.subheader("Research Q&A Summary:")
    for pair in st.session_state.qa_pairs:
        with st.expander(f"Q: {pair['question']}"):
            st.markdown(f"**A:** {pair['answer']}")

# --- Step 3: Compile Final Report ---
if st.session_state.qa_pairs and st.button("3. Compile Final Report into Google Doc"):
    openai_key_val = get_api_key("OPENAI", st.session_state)
    together_key_val = get_api_key("TOGETHERAI", st.session_state)
    composio_key_val = get_api_key("COMPOSIO", st.session_state)

    if not (openai_key_val or together_key_val):
        st.error("LLM API Key needed for report compilation.")
    elif not composio_key_val:
        st.error("Composio API Key needed to create Google Doc.")
    else:
        llm_instance = get_llm(openai_api_key=openai_key_val, together_api_key=together_key_val)
        if not llm_instance:
            st.stop()

        try:
            gdoc_tool = ComposioBasedGoogleDocTool(composio_api_key=composio_key_val)
        except Exception as e:
            st.error(f"Failed to initialize Google Doc tool: {e}")
            st.stop()

        report_compiler_agent = Agent(
            role='Expert Report Compiler',
            goal=f'Compile all research questions and their answers for the topic "{topic}" in domain "{domain}" into a coherent, well-structured research report. Then, save this report to a Google Document.',
            backstory='A professional editor and technical writer skilled at synthesizing complex information into clear reports.',
            llm=llm_instance,
            tools=[gdoc_tool],
            verbose=True,
            allow_delegation=False
        )

        qa_summary_for_report = "\n\n".join([f"Question: {p['question']}\nAnswer:\n{p['answer']}" for p in st.session_state.qa_pairs])
        report_title = f"Deep Dive Research Report: {topic} in {domain}"

        compile_task = Task(
            description=f'Compile the following research Q&A into a comprehensive report titled "{report_title}". The report should have an introduction, a main body organized by question/answer, and a brief conclusion. After generating the report content in markdown, use the ComposioBasedGoogleDocTool to create a Google Document with this content. The document name should be "{report_title}".\n\nResearch Data:\n{qa_summary_for_report}',
            expected_output=f'A confirmation message including the link to the Google Document containing the final report. Example: "Report compiled. Google Doc Link: http://docs.google.com/..."',
            agent=report_compiler_agent
        )

        temp_crew = Crew(agents=[report_compiler_agent], tasks=[compile_task], process=Process.sequential, verbose=0)

        with st.spinner("Compiling final report and creating Google Doc..."):
            try:
                report_result = temp_crew.kickoff(inputs={'qa_summary': qa_summary_for_report, 'report_title': report_title})
                st.session_state.final_report_link = report_result
                st.success("Final report compiled and Google Doc created!")
                st.markdown(f"**Final Report Status:** {st.session_state.final_report_link}")

                # Try to extract and display a clickable link
                if "http" in str(st.session_state.final_report_link):
                    link_start = str(st.session_state.final_report_link).find("http")
                    link_end_space = str(st.session_state.final_report_link).find(" ", link_start)
                    link_end_newline = str(st.session_state.final_report_link).find("\n", link_start)

                    if link_end_space == -1 and link_end_newline == -1:
                        actual_link = str(st.session_state.final_report_link)[link_start:]
                    elif link_end_space != -1 and (link_end_newline == -1 or link_end_space < link_end_newline):
                        actual_link = str(st.session_state.final_report_link)[link_start:link_end_space]
                    elif link_end_newline != -1:
                        actual_link = str(st.session_state.final_report_link)[link_start:link_end_newline]
                    else: # Should not happen if "http" is present
                        actual_link = None

                    if actual_link:
                        st.markdown(f"🔗 [Open Final Report Google Doc]({actual_link.strip()})")

            except Exception as e:
                st.error(f"Error compiling report or creating Google Doc: {e}")
                st.session_state.final_report_link = f"Error: {e}"
                import traceback
                st.text(traceback.format_exc())

st.markdown("---")
st.caption("Powered by CrewAI, Streamlit, Composio, and your chosen LLM/Search provider.")
