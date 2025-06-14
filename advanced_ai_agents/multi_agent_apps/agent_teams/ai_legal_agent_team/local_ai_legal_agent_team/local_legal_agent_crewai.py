import streamlit as st
import os
import tempfile
import re
from typing import Optional # Moved import here
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings # For local embeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama as LangchainOllamaLLM # For CrewAI if direct Ollama wrapper is not used/preferred

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from crewai.llms import Ollama # CrewAI's Ollama wrapper

# Qdrant client
import qdrant_client

load_dotenv()

# --- Configuration ---
# It's good practice to allow these to be overridden by environment variables or user input
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") # Assuming local Qdrant for a local setup

# --- Helper for API Keys/URLs (less emphasis on keys for local) ---
def get_config_value(config_key: str, session_state, default_value: str = ""):
    env_var_name = config_key.upper()
    if env_var_name in os.environ:
        return os.environ[env_var_name]
    return session_state.get(config_key.lower(), default_value)


# --- Document Processing with LangChain and Qdrant (Local Embeddings) ---
def process_uploaded_pdf_to_qdrant_local(uploaded_file,
                                         qdrant_url: str,
                                         qdrant_api_key: Optional[str], # Qdrant might still be secured
                                         ollama_base_url: str,
                                         ollama_embedding_model: str,
                                         collection_name: str):
    if not all([qdrant_url, ollama_base_url, ollama_embedding_model]):
        st.error("Qdrant URL, Ollama Base URL, and Ollama Embedding Model must be provided.")
        return None

    try:
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

        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_embedding_model
        )

        qdrant_client_instance = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None, # Handle optional API key
            timeout=60
        )

        vector_store = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            collection_name=collection_name,
            force_recreate=True,
        )
        st.success(f"PDF processed with local embeddings and stored in Qdrant collection: {collection_name}")
        os.unlink(tmp_file_path)
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDF for Qdrant (local): {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return None

# --- Custom CrewAI RAG Tool (Local Embeddings) ---
class LocalLegalDocumentRAGTool(BaseTool):
    name: str = "Local Legal Document RAG Tool"
    description: str = "Performs RAG from a legal document vector store using local embeddings."

    qdrant_url: str
    qdrant_api_key: Optional[str]
    collection_name: str
    ollama_base_url: str
    ollama_embedding_model: str
    vector_store: Qdrant = None

    def __init__(self, qdrant_url: str, collection_name: str, ollama_base_url: str, ollama_embedding_model: str, qdrant_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.ollama_embedding_model = ollama_embedding_model
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        if not self.vector_store:
            try:
                client = qdrant_client.QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=60)
                embeddings = OllamaEmbeddings(base_url=self.ollama_base_url, model=self.ollama_embedding_model)
                self.vector_store = Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embeddings=embeddings
                )
                print(f"RAG Tool: Qdrant vector store '{self.collection_name}' initialized successfully with local embeddings.")
            except Exception as e:
                print(f"RAG Tool: Failed to initialize Qdrant vector store with local embeddings: {e}")
                self.vector_store = None

    def _run(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return "Error: RAG Tool's Qdrant vector store (local) is not initialized."
        try:
            retrieved_docs = self.vector_store.similarity_search(query=query, k=k)
            if not retrieved_docs:
                return "No relevant information found in the document for your query (local search)."

            formatted_results = "\n\n---\n\n".join([f"Source Chunk (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in retrieved_docs])
            return f"Retrieved Information from Document (Local RAG):\n{formatted_results}"
        except Exception as e:
            return f"Error during local RAG similarity search: {e}"

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Local AI Legal Analysis Team (CrewAI)")
st.title("⚖️ Local AI Legal Analysis Team (CrewAI Version)")

with st.sidebar:
    st.header("Local LLM & Embeddings Configuration")
    st.session_state.ollama_base_url = st.text_input("Ollama Server URL", value=get_config_value("OLLAMA_BASE_URL", st.session_state, DEFAULT_OLLAMA_URL))
    st.session_state.ollama_llm_model = st.text_input("Ollama LLM Model Name", value=get_config_value("OLLAMA_LLM_MODEL", st.session_state, DEFAULT_LLM_MODEL))
    st.session_state.ollama_embedding_model = st.text_input("Ollama Embedding Model Name", value=get_config_value("OLLAMA_EMBEDDING_MODEL", st.session_state, DEFAULT_EMBEDDING_MODEL))

    st.header("Qdrant Configuration (Local or Remote)")
    st.session_state.qdrant_url = st.text_input("Qdrant URL", value=get_config_value("QDRANT_URL", st.session_state, DEFAULT_QDRANT_URL))
    st.session_state.qdrant_api_key = st.text_input("Qdrant API Key (Optional)", type="password", value=get_config_value("QDRANT_API_KEY", st.session_state, ""))


st.header("Document Upload & Processing (Local)")
uploaded_file = st.file_uploader("Upload a PDF document for local analysis", type="pdf")
COLLECTION_NAME_PREFIX = "local_legal_doc_"

if uploaded_file is not None:
    if 'current_local_collection_name' not in st.session_state or st.session_state.get('local_uploaded_file_name') != uploaded_file.name:
        st.session_state.local_uploaded_file_name = uploaded_file.name
        st.session_state.current_local_collection_name = COLLECTION_NAME_PREFIX + re.sub(r'\W+', '_', uploaded_file.name.split('.')[0].lower())

        q_url = get_config_value("qdrant_url", st.session_state, DEFAULT_QDRANT_URL)
        q_api_key = get_config_value("qdrant_api_key", st.session_state, "")
        ollama_url = get_config_value("ollama_base_url", st.session_state, DEFAULT_OLLAMA_URL)
        embed_model = get_config_value("ollama_embedding_model", st.session_state, DEFAULT_EMBEDDING_MODEL)

        if q_url and ollama_url and embed_model:
            with st.spinner(f"Processing PDF with local embeddings into Qdrant collection '{st.session_state.current_local_collection_name}'..."):
                vector_store_instance = process_uploaded_pdf_to_qdrant_local(
                    uploaded_file,
                    qdrant_url=q_url,
                    qdrant_api_key=q_api_key,
                    ollama_base_url=ollama_url,
                    ollama_embedding_model=embed_model,
                    collection_name=st.session_state.current_local_collection_name
                )
                if vector_store_instance:
                    st.session_state.local_vector_store_initialized = True
                else:
                    st.session_state.local_vector_store_initialized = False
                    st.error("Failed to initialize local vector store. Please check PDF and configurations.")
        else:
            st.warning("Qdrant URL, Ollama Server URL, and Embedding Model are required to process the PDF.")
            st.session_state.local_vector_store_initialized = False
elif 'local_uploaded_file_name' in st.session_state:
    del st.session_state.local_uploaded_file_name
    if 'current_local_collection_name' in st.session_state: del st.session_state.current_local_collection_name
    if 'local_vector_store_initialized' in st.session_state: del st.session_state.local_vector_store_initialized
    st.info("PDF deselected. Upload a new PDF for local analysis.")


st.header("Local Legal Analysis Query")
analysis_type = st.selectbox("Select Analysis Type (Local):", ["General Summary", "Risk Assessment", "Clause Identification"]) # Removed Compliance Check as it might need web search
user_query = st.text_area("Specific Question or Focus for Local Analysis:", placeholder="e.g., 'Identify all termination clauses.'")

if st.button("Analyze Document with Local CrewAI"):
    ollama_base_url = get_config_value("ollama_base_url", st.session_state, DEFAULT_OLLAMA_URL)
    ollama_llm_model = get_config_value("ollama_llm_model", st.session_state, DEFAULT_LLM_MODEL)
    ollama_embedding_model = get_config_value("ollama_embedding_model", st.session_state, DEFAULT_EMBEDDING_MODEL)
    qdrant_url = get_config_value("qdrant_url", st.session_state, DEFAULT_QDRANT_URL)
    qdrant_api_key = get_config_value("qdrant_api_key", st.session_state, "")


    if not all([ollama_base_url, ollama_llm_model, ollama_embedding_model]):
        st.error("Ollama Server URL, LLM Model, and Embedding Model must be configured.")
    elif not st.session_state.get('local_vector_store_initialized') or not st.session_state.get('current_local_collection_name'):
        st.error("Please upload and successfully process a PDF document with local embeddings first.")
    elif not user_query:
        st.error("Please enter a specific question or focus for the analysis.")
    else:
        # Instantiate RAG tool with local embedding settings
        local_rag_tool = LocalLegalDocumentRAGTool(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=st.session_state.current_local_collection_name,
            ollama_base_url=ollama_base_url,
            ollama_embedding_model=ollama_embedding_model
        )

        # Define Ollama LLM for CrewAI
        # Using CrewAI's Ollama wrapper
        local_llm = Ollama(model=ollama_llm_model, base_url=ollama_base_url)
        # Alternatively, using Langchain's OllamaLLM wrapper if more control is needed or CrewAI's isn't sufficient
        # local_llm = LangchainOllamaLLM(model=ollama_llm_model, base_url=ollama_base_url)


        # --- Define Local CrewAI Agents ---
        local_legal_researcher = Agent(
            role='Local Legal Researcher',
            goal=f'Thoroughly research and retrieve relevant information ONLY from the uploaded legal document regarding: "{user_query}" related to {analysis_type}.',
            backstory='An expert legal researcher focused on information retrieval solely from provided legal texts using local AI models.',
            tools=[local_rag_tool],
            llm=local_llm,
            verbose=True,
            allow_delegation=False
        )
        local_contract_analyst = Agent(
            role='Local Contract Analyst Specialist',
            goal=f'Analyze specific clauses and sections of the uploaded legal document based on the query "{user_query}" and analysis type "{analysis_type}", using only local AI models.',
            backstory='A meticulous analyst specializing in contract law, using local AI models to interpret legal language and identify key contractual elements from the provided document.',
            tools=[local_rag_tool],
            llm=local_llm,
            verbose=True,
            allow_delegation=False
        )
        local_legal_strategist = Agent(
            role='Local Legal Strategist and Advisor',
            goal=f'Synthesize research and analysis from the document to provide strategic advice, identify risks, and suggest actions based on the query "{user_query}" and analysis type "{analysis_type}", using local AI models.',
            backstory='A seasoned legal strategist providing actionable insights based on in-depth legal analysis of the document, powered by local AI.',
            tools=[local_rag_tool],
            llm=local_llm,
            verbose=True,
            allow_delegation=False
        )

        # --- Define Local CrewAI Tasks ---
        research_task = Task(
            description=f'Conduct comprehensive research on the query: "{user_query}". Focus on {analysis_type}. Prioritize information from the uploaded document using the LocalLegalDocumentRAGTool.',
            expected_output=f'A detailed report summarizing all findings, citing document sections for the query "{user_query}" concerning {analysis_type}.',
            agent=local_legal_researcher
        )
        analysis_task = Task(
            description=f'Perform a detailed analysis of the document concerning "{user_query}" and {analysis_type}. Extract relevant clauses, terms, and conditions. Utilize the LocalLegalDocumentRAGTool extensively.',
            expected_output=f'A specific analysis of the document sections relevant to "{user_query}" ({analysis_type}), highlighting key legal points and interpretations.',
            agent=local_contract_analyst,
            context=[research_task]
        )
        strategy_task = Task(
            description=f'Develop legal strategies or advice based on the research and analysis for "{user_query}" ({analysis_type}). Identify potential risks or actionable recommendations from the document.',
            expected_output=f'A strategic report outlining advice, risks, and recommendations pertaining to "{user_query}" ({analysis_type}), based solely on the provided document.',
            agent=local_legal_strategist,
            context=[analysis_task]
        )

        # --- Define Local Crew ---
        local_legal_crew = Crew(
            agents=[local_legal_researcher, local_contract_analyst, local_legal_strategist],
            tasks=[research_task, analysis_task, strategy_task],
            process=Process.sequential,
            verbose=True
        )

        st.info("⚖️ Local Legal CrewAI is analyzing your document... This may take some time.")

        crew_inputs = {
            'user_query': user_query,
            'analysis_type': analysis_type,
            'document_collection': st.session_state.current_local_collection_name
        }

        try:
            crew_result = local_legal_crew.kickoff(inputs=crew_inputs)

            st.subheader("📜 Local Legal Analysis Results:")
            st.markdown("---")
            st.markdown(crew_result)

        except Exception as e:
            st.error(f"Error during Local CrewAI legal analysis: {e}")
            st.error(f"Ollama URL: {ollama_base_url}, LLM: {ollama_llm_model}, Embed Model: {ollama_embedding_model}")
            st.error(f"Qdrant URL: {qdrant_url}, Qdrant API Key set: {bool(qdrant_api_key)}")
            st.error(f"Current Collection: {st.session_state.get('current_local_collection_name')}")
            import traceback
            st.text(traceback.format_exc())


st.markdown("---")
st.caption("Powered by Local CrewAI, LangChain (Ollama), Qdrant, and Streamlit.")
