# ##################################################################################################
# # DEPRECATION NOTE:
# # This file is deprecated. Please use research_agent_llama3_crewai.py for the CrewAI-based
# # implementation of the Hacker News Research Agent Team using Llama3 with Ollama.
# ##################################################################################################

# Import the required libraries
import streamlit as st
from agno.agent import Agent
from agno.tools.hackernews import HackerNews
from agno.models.ollama import Ollama

# Set up the Streamlit app
st.title("Multi-Agent AI Researcher using Llama-3 🔍🤖")
st.caption("This app allows you to research top stories and users on HackerNews and write blogs, reports and social posts.")

# Create instances of the Assistant
story_researcher = Agent(
    name="HackerNews Story Researcher",
    role="Researches hackernews stories and users.",
    tools=[HackerNews()],
    model=Ollama(id="llama3.2", max_tokens=1024)
)

user_researcher = Agent(
    name="HackerNews User Researcher",
    role="Reads articles from URLs.",
    tools=[HackerNews()],
    model=Ollama(id="llama3.2", max_tokens=1024)
)

hn_assistant = Agent(
    name="Hackernews Team",
    team=[story_researcher, user_researcher],
    model=Ollama(id="llama3.2", max_tokens=1024)
)

# Input field for the report query
query = st.text_input("Enter your report query")

if query:
    # Get the response from the assistant
    response = hn_assistant.run(query, stream=False)
    st.write(response.content)