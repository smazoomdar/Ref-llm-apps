import streamlit as st
import os
from PIL import Image
import io
import base64 # Moved import here
from typing import Optional, Dict, Any # Moved typing imports here
from dotenv import load_dotenv

# E2B Code Interpreter
from e2b import Sandbox as E2BSandbox # Renamed to avoid conflict with a potential Sandbox class

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from crewai.llms import ChatOpenAI, ChatGoogleGenerativeAI # Using CrewAI's built-in wrappers

load_dotenv()

# --- Helper for API Keys ---
def get_api_key(service_name: str, session_state, default_value: str = ""):
    env_var_key = f"{service_name.upper()}_API_KEY"
    env_var_value = os.getenv(env_var_key)
    if env_var_value:
        return env_var_value
    return session_state.get(f"{service_name.lower()}_api_key", default_value)

# --- E2B Sandbox Management ---
def initialize_e2b_sandbox(api_key: str) -> Optional[E2BSandbox]:
    if not api_key:
        st.error("E2B API Key is required to initialize the sandbox.")
        return None
    try:
        # Note: E2BSandbox is stateful. A new one is created each time, or manage lifecycle.
        # For this app, if each "Analyze" click should be a fresh environment, this is okay.
        # If state needs to persist across tool calls within a single Crew run, the sandbox
        # instance needs to be managed and passed around or held by the tool instance.
        sandbox = E2BSandbox(api_key=api_key)
        # sandbox.open() # Not needed for current e2b versions, connection is implicit
        st.success("E2B Sandbox initialized (or re-initialized).")
        return sandbox
    except Exception as e:
        st.error(f"Failed to initialize E2B Sandbox: {e}")
        return None

def close_e2b_sandbox(sandbox: Optional[E2BSandbox]):
    if sandbox:
        try:
            sandbox.close()
            st.info("E2B Sandbox closed.")
        except Exception as e:
            st.warning(f"Error closing E2B Sandbox: {e}")


# --- Custom E2B Code Execution Tool ---
class E2BCodeExecutionTool(BaseTool):
    name: str = "E2B Code Execution Tool"
    description: str = "Executes Python code in a secure E2B sandbox and returns the results."
    sandbox: Optional[E2BSandbox] = None # Can be passed in or managed by the tool

    # Option 1: Pass sandbox instance during __init__ (preferred if managed by Streamlit)
    def __init__(self, sandbox_instance: Optional[E2BSandbox] = None, e2b_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if sandbox_instance:
            self.sandbox = sandbox_instance
        elif e2b_api_key: # Fallback: tool manages its own sandbox (less ideal for shared state)
            try:
                self.sandbox = E2BSandbox(api_key=e2b_api_key)
                print("E2BCodeExecutionTool: Initialized its own E2B sandbox.")
            except Exception as e:
                print(f"E2BCodeExecutionTool: Failed to initialize its own sandbox: {e}")
                self.sandbox = None
        else:
            print("E2BCodeExecutionTool: Warning - No E2B sandbox instance or API key provided at init.")


    def _run(self, code: str) -> Dict[str, Any]:
        if not self.sandbox:
            return {"error": "E2B Sandbox is not available to the tool.", "stdout": "", "stderr": "", "files": []}

        try:
            # Ensure the sandbox is connected; e2b-code-interpreter handles this implicitly on first command
            # For older e2b versions, you might need self.sandbox.open() if not already open.

            execution = self.sandbox.notebook.exec_cell(code) # exec_cell is simpler for script-like execution

            # Retrieve files (example: list all files in current directory)
            # This part needs to be adapted based on what files are expected/needed.
            # For simplicity, let's list files in /home/user or try to get specific output files if known.
            # sandbox_files = self.sandbox.fs.ls("/home/user") # Example listing
            # For now, let's assume no specific file output is captured unless explicitly written by the code.
            # If code writes to a known file, use self.sandbox.download_file(...)

            # The result from exec_cell includes stdout, stderr, and potentially rich outputs like images
            # For simplicity, we'll focus on text-based stdout/stderr and errors.

            return {
                "stdout": "".join(out.text for out in execution.outputs if out.name == "stdout"),
                "stderr": "".join(out.text for out in execution.outputs if out.name == "stderr"),
                "error": execution.error.name if execution.error else None,
                # "files": [f.name for f in sandbox_files] if sandbox_files else []
                # Files part needs more specific logic if we want to return file contents or names.
                # For now, if the code generates a file, it's in the sandbox but not directly returned here.
                "execution_outputs": [out.text for out in execution.outputs] # Raw output list
            }
        except Exception as e:
            return {"error": f"Error during code execution in E2B: {e}", "stdout": "", "stderr": "", "files": []}

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Multimodal Coding Team (CrewAI)")
st.title("✨ AI Multimodal Coding Agent Team (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state))
    st.session_state.google_gemini_api_key = st.text_input("Google Gemini API Key", type="password", value=get_api_key("GOOGLE_GEMINI", st.session_state))
    st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password", value=get_api_key("E2B", st.session_state))

st.header("Provide Input (Image or Text Query)")
uploaded_image = st.file_uploader("Upload an image (e.g., UI mockup, diagram)", type=["png", "jpg", "jpeg"])
user_text_query = st.text_area("Or, enter a text-based coding problem:", placeholder="e.g., 'Write a Python script to parse a CSV file and calculate the average of a column.'")

if 'e2b_sandbox' not in st.session_state:
    st.session_state.e2b_sandbox = None


if st.button("Generate Code with CrewAI"):
    openai_key = get_api_key("OPENAI", st.session_state)
    gemini_key = get_api_key("GOOGLE_GEMINI", st.session_state)
    e2b_key = get_api_key("E2B", st.session_state)

    if not openai_key or not e2b_key: # Gemini is optional if only text query
        st.error("OpenAI and E2B API Keys are required.")
    elif not uploaded_image and not user_text_query:
        st.error("Please upload an image or provide a text query.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        # For Gemini, CrewAI's ChatGoogleGenerativeAI uses GOOGLE_API_KEY
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key

        problem_description = None

        # Initialize E2B Sandbox if not already done or if it needs to be fresh
        # For simplicity, we re-initialize here. In a more complex app, you might want to keep it open.
        if st.session_state.e2b_sandbox: # Close previous if any
            close_e2b_sandbox(st.session_state.e2b_sandbox)
        st.session_state.e2b_sandbox = initialize_e2b_sandbox(e2b_key)

        if not st.session_state.e2b_sandbox:
            st.error("Failed to initialize E2B Sandbox. Cannot proceed with code execution.")
            st.stop()

        code_execution_tool = E2BCodeExecutionTool(sandbox_instance=st.session_state.e2b_sandbox)

        # Initialize LLMs for CrewAI
        openai_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2) # Or gpt-3.5-turbo
        gemini_llm = None
        if gemini_key:
            try:
                gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=gemini_key)
            except Exception as e:
                st.warning(f"Could not initialize Gemini LLM (gemini-pro-vision might need specific setup or model name 'gemini-1.5-flash-latest'): {e}. Image processing will be skipped if Gemini is not available.")


        if uploaded_image is not None:
            if not gemini_key or not gemini_llm:
                st.warning("Google Gemini API Key not provided or LLM init failed. Skipping image processing.")
            else:
                with st.spinner("Vision Agent is analyzing the image..."):
                    try:
                        image_bytes = uploaded_image.getvalue()
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Vision Agent
                        vision_agent = Agent(
                            role='Vision Problem Extractor',
                            goal='Analyze the uploaded image (e.g., UI mockup, diagram, handwritten note) and describe the coding problem or task it represents in clear, detailed text.',
                            backstory='An AI expert in interpreting visual information and translating it into actionable development tasks.',
                            llm=gemini_llm, # Uses Gemini for vision
                            verbose=True,
                            allow_delegation=False
                        )

                        # Task for Vision Agent
                        # CrewAI's ChatGoogleGenerativeAI might need specific input format for images.
                        # Often, it's a list of content blocks (text + image).
                        # The exact way to pass image to Gemini via CrewAI's wrapper needs care.
                        # For now, let's assume the LLM can handle an image if passed via context or a special field.
                        # A common pattern is `human_message_content = [{"type": "text", "text": "prompt"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`
                        # This part is complex with CrewAI's current abstractions for multimodal.
                        # For now, we'll pass the image path/bytes conceptually and hope the wrapper handles it or use a placeholder.

                        # Simplified approach: Pass image description as a placeholder if direct image passing is tricky
                        # In a real scenario, you'd ensure the LLM call includes the image data.
                        # For this example, let's assume the vision agent's LLM (Gemini) is configured to accept image data
                        # and the task description is enough to guide it.
                        # The `kickoff` input might need to be structured for multimodal.

                        # Let's use a placeholder for the image content in the task description for now.
                        # The actual image data would need to be handled by the ChatGoogleGenerativeAI integration.
                        vision_task_description = f"Analyze the provided image (conceptually representing '{uploaded_image.name}') and extract a detailed coding problem. Describe UI elements, interactions, data flow, or any logic implied by the image."

                        # Create a temporary crew for the vision task
                        # This task might need a special way to receive image input.
                        # For now, let's assume the LLM used by vision_agent can be directly invoked with image.
                        # This part is highly dependent on CrewAI's specific multimodal support with Gemini.
                        # A direct call to Gemini might be easier here than via CrewAI task if image passing is complex.

                        # Alternative for Gemini Vision with CrewAI - pass it in the kickoff / task input if supported
                        # For now, we'll simulate this by just getting a text description.
                        # This is a placeholder for actual multimodal call.
                        st.warning("Note: True multimodal image analysis with Gemini via CrewAI task is complex. This part is simplified.")

                        # Simplified: Assume Gemini can describe the image if given a prompt.
                        # This is not how CrewAI tasks typically work with multimodal directly.
                        # A more robust solution would use a custom tool or direct LLM call for vision.
                        if hasattr(gemini_llm, 'invoke') or hasattr(gemini_llm, '_make_api_request_model_garden'): # Check if it's a Langchain LLM
                            from langchain_core.messages import HumanMessage
                            response = gemini_llm.invoke([HumanMessage(content=[
                                {"type": "text", "text": "Describe the coding problem shown in this image. Be detailed about UI, logic, and desired output."},
                                {"type": "image_url", "image_url": f"data:{uploaded_image.type};base64,{base64.b64encode(image_bytes).decode()}"}
                            ])])
                            problem_description = response.content
                            st.info("Image analysis by Gemini (direct call simulation):")
                            st.caption(problem_description)
                        else:
                            st.error("Gemini LLM via CrewAI does not support direct invoke for multimodal in this simplified test. Problem description from image will be empty.")
                            problem_description = "Error: Could not process image with Gemini via simplified test."


                    except Exception as e:
                        st.error(f"Error during image analysis: {e}")
                        problem_description = f"Error processing image: {e}"
                        import traceback
                        st.text(traceback.format_exc())

        elif user_text_query:
            problem_description = user_text_query

        if problem_description:
            st.subheader("Problem Description for Coding Agents:")
            st.markdown(problem_description)

            with st.spinner("Coding Agents are working..."):
                # Define Coding and Execution Agents
                python_coder = Agent(
                    role='Python Coding Specialist',
                    goal='Write clean, efficient, and correct Python code to solve the given problem_description. The code should be complete and runnable.',
                    backstory='An expert Python developer with a strong background in various application domains. Focuses on creating production-quality code.',
                    llm=openai_llm, # Uses OpenAI
                    verbose=True,
                    allow_delegation=False
                )
                code_analyzer = Agent(
                    role='Code Execution and Analysis Agent',
                    goal='Execute the provided Python code in a sandbox, analyze its output, identify any errors, and suggest improvements or confirm success.',
                    backstory='A meticulous QA engineer and code reviewer who uses a sandboxed environment to test code thoroughly and provide feedback.',
                    llm=openai_llm, # Uses OpenAI
                    tools=[code_execution_tool],
                    verbose=True,
                    allow_delegation=False
                )

                # Define Tasks for Coding and Execution
                coding_task = Task(
                    description=f'Based on the following problem description, write a complete Python script:\n\n"{problem_description}"\n\nEnsure the code is well-commented and handles potential edge cases if appropriate. Output only the raw Python code, without any markdown formatting or explanations before or after the code block.',
                    expected_output='The raw Python code string. For example: "print(\'Hello World\')" or "def my_func():\\n  pass"',
                    agent=python_coder
                )
                execution_task = Task(
                    description='Take the Python code generated by the Python Coding Specialist (from context). Execute this code using the E2BCodeExecutionTool. Analyze the results (stdout, stderr, errors, files). If there are errors, explain them and suggest fixes. If successful, confirm and summarize the output.',
                    expected_output='A JSON object containing: "execution_stdout", "execution_stderr", "execution_error" (if any), "analysis_and_feedback" (string). If code needed fixing and was fixed and re-run by you, provide analysis of the final successful run.',
                    agent=code_analyzer,
                    context=[coding_task] # Depends on the coding_task's output
                )

                # Create and Kickoff Main Crew
                coding_crew = Crew(
                    agents=[python_coder, code_analyzer],
                    tasks=[coding_task, execution_task],
                    process=Process.sequential,
                    verbose=True
                )

                crew_result = coding_crew.kickoff(inputs={'problem_description': problem_description}) # This input is for the first task implicitly

                st.subheader("CrewAI Coding and Execution Analysis:")
                st.markdown("---")
                # Displaying the raw code from the coding_task
                if coding_task.output:
                    st.markdown("#### Generated Code:")
                    st.code(coding_task.output.exported_output, language="python") # Assuming raw code string
                else:
                    st.warning("No code output from Python Coding Agent.")

                st.markdown("#### Execution Analysis (from CodeExecutionAndAnalysisAgent):")
                st.json(crew_result) # The result of the last task (execution_task)

        # Close sandbox after use (important!)
        close_e2b_sandbox(st.session_state.e2b_sandbox)
        st.session_state.e2b_sandbox = None # Reset for next run
