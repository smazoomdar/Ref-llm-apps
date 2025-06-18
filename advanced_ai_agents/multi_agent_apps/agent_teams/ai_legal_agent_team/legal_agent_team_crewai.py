import streamlit as st
import os
import tempfile
import re # Moved import re to the top
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, DuckDuckGoSearchRun

# Qdrant client
import qdrant_client

load_dotenv()

# --- API Key Management ---
def get_api_key(service_name: str, session_state):
    env_var = f"{service_name.upper()}_API_KEY"
    if env_var in os.environ:
        return os.environ[env_var]
    return session_state.get(f"{service_name.lower()}_api_key")

# --- Document Processing with LangChain and Qdrant ---
def process_uploaded_pdf_to_qdrant(uploaded_file, qdrant_url: str, qdrant_api_key: str, openai_api_key: str, collection_name: str):
    if not all([qdrant_url, qdrant_api_key, openai_api_key]):
        st.error("Qdrant URL, Qdrant API Key, and OpenAI API Key must be provided for document processing.")
        return None

    try:
        # Save uploaded file to a temporary path for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        if not documents:
            st.error("Could not load any documents from the PDF.")
            os.unlink(tmp_file_path)
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        if not texts:
            st.error("Could not split the document into text chunks.")
            os.unlink(tmp_file_path)
            return None

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        # Initialize Qdrant client
        qdrant_client_instance = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60 # Increased timeout
        )

        # Check if collection exists, if not, Qdrant.from_documents will create it.
        # Forcing recreation for simplicity in this example, or use a unique name per upload.
        # In a production scenario, you might want to manage collections more carefully.

        vector_store = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=True, # Recreate collection for each new PDF upload for simplicity
        )
        st.success(f"PDF processed and embeddings stored in Qdrant collection: {collection_name}")
        os.unlink(tmp_file_path) # Clean up temporary file
        return vector_store # Or return collection_name for later use

    except Exception as e:
        st.error(f"Error processing PDF for Qdrant: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return None

# --- Custom CrewAI RAG Tool ---
class LegalDocumentRAGTool(BaseTool):
    name: str = "Legal Document RAG Tool"
    description: str = "Performs Retrieval Augmented Generation from a specific legal document vector store."

    qdrant_url: str
    qdrant_api_key: str
    collection_name: str
    openai_api_key: str
    vector_store: Qdrant = None

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str, openai_api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        if not self.vector_store:
            try:
                client = qdrant_client.QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=60)
                embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
                self.vector_store = Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embeddings=embeddings
                )
            except Exception as e:
                # st.error(f"RAG Tool: Failed to initialize Qdrant vector store: {e}") # Cannot use st here
                print(f"RAG Tool: Failed to initialize Qdrant vector store: {e}")
                self.vector_store = None

    def _run(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return "Error: RAG Tool's Qdrant vector store is not initialized. Cannot perform search."
        try:
            retrieved_docs = self.vector_store.similarity_search(query=query, k=k)
            if not retrieved_docs:
                return "No relevant information found in the document for your query."

            formatted_results = "\n\n---\n\n".join([f"Source Chunk (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in retrieved_docs])
            return f"Retrieved Information from Document:\n{formatted_results}"
        except Exception as e:
            return f"Error during RAG similarity search: {e}"

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="AI Legal Analysis Team (CrewAI)")
st.title("⚖️ AI Legal Analysis Team (CrewAI Version)")

with st.sidebar:
    st.header("API Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=get_api_key("OPENAI", st.session_state) or "")
    st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", type="password", value=get_api_key("QDRANT", st.session_state) or "")
    st.session_state.qdrant_url = st.text_input("Qdrant URL", value=get_api_key("QDRANT_URL", st.session_state) or os.getenv("QDRANT_URL") or "http://localhost:6333")
    st.session_state.duckduckgo_api_key = st.text_input("DuckDuckGo API Key (Optional)", type="password", value=get_api_key("DUCKDUCKGO", st.session_state) or "")


st.header("Document Upload & Processing")
uploaded_file = st.file_uploader("Upload a PDF document for analysis", type="pdf")
COLLECTION_NAME_PREFIX = "legal_doc_"

if uploaded_file is not None:
    if 'current_collection_name' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        # Create a unique collection name for each uploaded file to avoid conflicts
        # For simplicity, using a hash of the file name or just a timestamped name
        # This demo will overwrite if the same file name is used. A better approach is needed for prod.
        st.session_state.current_collection_name = COLLECTION_NAME_PREFIX + re.sub(r'\W+', '_', uploaded_file.name.split('.')[0].lower())

        q_url = get_api_key("QDRANT_URL", st.session_state) or os.getenv("QDRANT_URL") or "http://localhost:6333"
        q_api_key = get_api_key("QDRANT", st.session_state)
        oai_api_key = get_api_key("OPENAI", st.session_state)

        if oai_api_key and q_api_key and q_url:
            with st.spinner(f"Processing PDF and building vector store in Qdrant collection '{st.session_state.current_collection_name}'..."):
                vector_store_instance = process_uploaded_pdf_to_qdrant(
                    uploaded_file,
                    qdrant_url=q_url,
                    qdrant_api_key=q_api_key,
                    openai_api_key=oai_api_key,
                    collection_name=st.session_state.current_collection_name
                )
                if vector_store_instance:
                    st.session_state.vector_store_initialized = True
                    # No need to store the whole instance, just the name and params for RAG tool
                else:
                    st.session_state.vector_store_initialized = False
                    st.error("Failed to initialize vector store. Please check PDF and API keys.")
        else:
            st.warning("OpenAI API Key, Qdrant API Key, and Qdrant URL are required to process the PDF.")
            st.session_state.vector_store_initialized = False
elif 'uploaded_file_name' in st.session_state: # File was deselected
    del st.session_state.uploaded_file_name
    if 'current_collection_name' in st.session_state: del st.session_state.current_collection_name
    if 'vector_store_initialized' in st.session_state: del st.session_state.vector_store_initialized
    st.info("PDF deselected. Upload a new PDF to enable analysis.")


st.header("Legal Analysis Query")
analysis_type = st.selectbox("Select Analysis Type:", ["General Summary", "Risk Assessment", "Clause Identification", "Compliance Check"])
user_query = st.text_area("Specific Question or Focus for Analysis:", placeholder="e.g., 'Identify all termination clauses in this contract.' or 'Are there any data privacy risks related to GDPR?'")

if st.button("Analyze Document with CrewAI"):
    openai_api_key = get_api_key("OPENAI", st.session_state)
    qdrant_api_key = get_api_key("QDRANT", st.session_state)
    qdrant_url = get_api_key("QDRANT_URL", st.session_state) or os.getenv("QDRANT_URL") or "http://localhost:6333"
    # duckduckgo_api_key = get_api_key("DUCKDUCKGO", st.session_state) # For web search agent

    if not openai_api_key:
        st.error("OpenAI API Key is required.")
    elif not st.session_state.get('vector_store_initialized') or not st.session_state.get('current_collection_name'):
        st.error("Please upload and successfully process a PDF document first.")
    elif not user_query:
        st.error("Please enter a specific question or focus for the analysis.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key # Set for CrewAI and Langchain Embeddings

        # Instantiate RAG tool
        rag_tool = LegalDocumentRAGTool(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=st.session_state.current_collection_name,
            openai_api_key=openai_api_key
        )

        # Instantiate Search tool (optional, based on agent needs)
        search_tool = DuckDuckGoSearchRun()


        # --- Define CrewAI Agents ---
        legal_researcher = Agent(
            role='Legal Researcher',
            goal=f'Thoroughly research and retrieve relevant information from the uploaded legal document and the web regarding: "{user_query}" related to {analysis_type}.',
            backstory='An expert legal researcher adept at information retrieval from legal texts and online sources. Focuses on factual accuracy and comprehensiveness.',
            tools=[rag_tool, search_tool],
            verbose=True,
            allow_delegation=False # For simplicity in this example
        )
        contract_analyst = Agent(
            role='Contract Analyst Specialist',
            goal=f'Analyze specific clauses and sections of the uploaded legal document based on the query "{user_query}" and analysis type "{analysis_type}".',
            backstory='A meticulous analyst specializing in contract law, with a keen eye for detail in interpreting legal language and identifying key contractual elements.',
            tools=[rag_tool],
            verbose=True,
            allow_delegation=False
        )
        legal_strategist = Agent(
            role='Legal Strategist and Advisor',
            goal=f'Synthesize research and analysis to provide strategic advice, identify risks, and suggest actions based on the query "{user_query}" and analysis type "{analysis_type}".',
            backstory='A seasoned legal strategist who provides actionable insights and recommendations based on in-depth legal analysis.',
            tools=[rag_tool], # May also need search_tool depending on strategy
            verbose=True,
            allow_delegation=False
        )

        # --- Define CrewAI Tasks ---
        research_task = Task(
            description=f'Conduct comprehensive research on the query: "{user_query}". Focus on {analysis_type}. Prioritize information from the uploaded document using the LegalDocumentRAGTool. Supplement with web research if necessary.',
            expected_output=f'A detailed report summarizing all findings, citing document sections and web sources for the query "{user_query}" concerning {analysis_type}.',
            agent=legal_researcher
        )
        analysis_task = Task(
            description=f'Perform a detailed analysis of the document concerning "{user_query}" and {analysis_type}. Extract relevant clauses, terms, and conditions. Utilize the LegalDocumentRAGTool extensively.',
            expected_output=f'A specific analysis of the document sections relevant to "{user_query}" ({analysis_type}), highlighting key legal points and interpretations.',
            agent=contract_analyst,
            context=[research_task] # Depends on research
        )
        strategy_task = Task(
            description=f'Develop legal strategies or advice based on the research and analysis for "{user_query}" ({analysis_type}). Identify potential risks, compliance issues, or actionable recommendations.',
            expected_output=f'A strategic report outlining advice, risks, and recommendations pertaining to "{user_query}" ({analysis_type}), based on the provided document and research.',
            agent=legal_strategist,
            context=[analysis_task] # Depends on analysis
        )

        # --- Define Crew ---
        legal_crew = Crew(
            agents=[legal_researcher, contract_analyst, legal_strategist],
            tasks=[research_task, analysis_task, strategy_task],
            process=Process.sequential,
            verbose=True
        )

        st.info("⚖️ Legal CrewAI is analyzing your document... This may take some time.")

        crew_inputs = { # These are implicitly used by agent/task descriptions via f-strings
            'user_query': user_query,
            'analysis_type': analysis_type,
            'document_collection': st.session_state.current_collection_name
        }

        try:
            crew_result = legal_crew.kickoff(inputs=crew_inputs)

            st.subheader("📜 Legal Analysis Results:")
            st.markdown("---")
            st.markdown(crew_result) # Display raw output of the last task

        except Exception as e:
            st.error(f"Error during CrewAI legal analysis: {e}")
            st.error(f"OpenAI API Key set: {bool(openai_api_key)}")
            st.error(f"Qdrant URL: {qdrant_url}, Qdrant API Key set: {bool(qdrant_api_key)}")
            st.error(f"Current Collection: {st.session_state.get('current_collection_name')}")
            import traceback
            st.text(traceback.format_exc())


st.markdown("---")
st.caption("Powered by CrewAI, LangChain, Qdrant, and Streamlit.")

# Removed import re from the bottom as it's now at the top
