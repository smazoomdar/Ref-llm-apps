import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Import necessary classes from the main script
from advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai import (
    LocalLegalDocumentRAGTool,
    process_uploaded_pdf_to_qdrant_local
)

# Mock streamlit for testing process_uploaded_pdf_to_qdrant_local's st.error/success calls
class MockStreamlitLocal:
    def error(self, message):
        print(f"Streamlit Error (Local Test): {message}")
    def success(self, message):
        print(f"Streamlit Success (Local Test): {message}")
    def spinner(self, text):
        return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

mock_st_local = MockStreamlitLocal()


class TestLocalLegalDocumentRAGTool(unittest.TestCase):

    def setUp(self):
        self.dummy_qdrant_url = "http://dummy_url_local:6333"
        self.dummy_qdrant_api_key = None # Typically None for fully local Qdrant
        self.dummy_collection_name = "test_local_collection"
        self.dummy_ollama_url = "http://localhost:11434"
        self.dummy_ollama_embedding_model = "nomic-embed-text"

        with patch.object(LocalLegalDocumentRAGTool, '_initialize_vector_store', return_value=None):
            self.tool = LocalLegalDocumentRAGTool(
                qdrant_url=self.dummy_qdrant_url,
                qdrant_api_key=self.dummy_qdrant_api_key,
                collection_name=self.dummy_collection_name,
                ollama_base_url=self.dummy_ollama_url,
                ollama_embedding_model=self.dummy_ollama_embedding_model
            )
            self.tool.vector_store = MagicMock() # Mock the vector_store directly

    def test_rag_tool_run_success_local(self):
        mock_doc = Document(page_content='Local test content', metadata={'page': 1})
        self.tool.vector_store.similarity_search.return_value = [mock_doc]

        result = self.tool._run("test query local", k=1)

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query local", k=1)
        self.assertIn("Retrieved Information from Document (Local RAG):", result)
        self.assertIn("Local test content", result)
        self.assertIn("Page 1", result)

    def test_rag_tool_run_no_results_local(self):
        self.tool.vector_store.similarity_search.return_value = []

        result = self.tool._run("test query no results local")

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query no results local", k=3)
        self.assertEqual(result, "No relevant information found in the document for your query (local search).")

    def test_rag_tool_initialization_failure_local(self):
        with patch.object(LocalLegalDocumentRAGTool, '_initialize_vector_store', return_value=None):
            tool_no_vs = LocalLegalDocumentRAGTool(
                qdrant_url=self.dummy_qdrant_url,
                qdrant_api_key=self.dummy_qdrant_api_key,
                collection_name=self.dummy_collection_name,
                ollama_base_url=self.dummy_ollama_url,
                ollama_embedding_model=self.dummy_ollama_embedding_model
            )
            tool_no_vs.vector_store = None

        result = tool_no_vs._run("test query local fail")
        self.assertEqual(result, "Error: RAG Tool's Qdrant vector store (local) is not initialized.")

    def test_rag_tool_similarity_search_exception_local(self):
        self.tool.vector_store.similarity_search.side_effect = Exception("Local similarity search failed")

        result = self.tool._run("test query local exception")

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query local exception", k=3)
        self.assertEqual(result, "Error during local RAG similarity search: Local similarity search failed")


@patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.st', mock_st_local)
class TestProcessUploadedPDFToQdrantLocal(unittest.TestCase):

    def setUp(self):
        self.mock_uploaded_file = MagicMock()
        self.mock_uploaded_file.getvalue.return_value = b"dummy local pdf content"
        self.qdrant_url = "http://test-qdrant-local.com"
        self.qdrant_api_key = None # Often None for local
        self.ollama_url = "http://localhost:11434"
        self.ollama_embedding_model = "mxbai-embed-large"
        self.collection_name = "test_local_pdf_collection"

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.os.unlink')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.tempfile.NamedTemporaryFile')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.Qdrant.from_documents')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.OllamaEmbeddings')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.RecursiveCharacterTextSplitter')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.PyPDFLoader')
    def test_process_pdf_success_local(self, MockPyPDFLoader, MockTextSplitter, MockOllamaEmbeddings, MockQdrantFromDocs, MockNamedTempFile, MockOsUnlink):
        # Configure mocks
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "dummy/temp/local_path.pdf"
        MockNamedTempFile.return_value.__enter__.return_value = mock_tmp_file

        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.return_value = [Document(page_content="local doc content")]

        mock_splitter_instance = MockTextSplitter.return_value
        mock_texts = [Document(page_content="local_chunk1")]
        mock_splitter_instance.split_documents.return_value = mock_texts

        mock_ollama_embeddings_instance = MockOllamaEmbeddings.return_value
        mock_qdrant_instance = MagicMock()
        MockQdrantFromDocs.return_value = mock_qdrant_instance

        with patch.object(mock_st_local, 'success') as mock_st_success_call:
            result = process_uploaded_pdf_to_qdrant_local(
                self.mock_uploaded_file, self.qdrant_url, self.qdrant_api_key,
                self.ollama_url, self.ollama_embedding_model, self.collection_name
            )

        MockNamedTempFile.assert_called_once_with(delete=False, suffix=".pdf")
        mock_tmp_file.write.assert_called_once_with(b"dummy local pdf content")
        MockPyPDFLoader.assert_called_once_with(mock_tmp_file.name)
        mock_loader_instance.load.assert_called_once()
        MockTextSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_splitter_instance.split_documents.assert_called_once_with([Document(page_content="local doc content")])
        MockOllamaEmbeddings.assert_called_once_with(base_url=self.ollama_url, model=self.ollama_embedding_model)
        MockQdrantFromDocs.assert_called_once_with(
            mock_texts,
            mock_ollama_embeddings_instance,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            force_recreate=True
        )
        MockOsUnlink.assert_called_once_with(mock_tmp_file.name)
        self.assertEqual(result, mock_qdrant_instance)
        mock_st_success_call.assert_called_once()


    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.os.unlink')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.tempfile.NamedTemporaryFile')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.local_ai_legal_agent_team.local_legal_agent_crewai.PyPDFLoader')
    def test_process_pdf_loader_fails_local(self, MockPyPDFLoader, MockNamedTempFile, MockOsUnlink):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "dummy/temp/local_path.pdf"
        MockNamedTempFile.return_value.__enter__.return_value = mock_tmp_file

        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.return_value = []

        with patch.object(mock_st_local, 'error') as mock_st_error_call:
            result = process_uploaded_pdf_to_qdrant_local(
                self.mock_uploaded_file, self.qdrant_url, self.qdrant_api_key,
                self.ollama_url, self.ollama_embedding_model, self.collection_name
            )
            self.assertIsNone(result)
            mock_st_error_call.assert_any_call("Could not load any documents from the PDF.")
        MockOsUnlink.assert_called_once_with(mock_tmp_file.name)

if __name__ == "__main__":
    unittest.main()
