import unittest
from unittest.mock import patch, MagicMock, mock_open
from langchain_core.documents import Document

# Import necessary classes from the main script
from advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai import (
    LegalDocumentRAGTool,
    process_uploaded_pdf_to_qdrant
)

# Mock streamlit for testing process_uploaded_pdf_to_qdrant's st.error calls
class MockStreamlit:
    def error(self, message):
        print(f"Streamlit Error: {message}") # Or raise an exception to catch in tests
    def success(self, message):
        print(f"Streamlit Success: {message}")
    def spinner(self, text):
        return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

mock_st = MockStreamlit()

class TestLegalDocumentRAGTool(unittest.TestCase):

    def setUp(self):
        self.dummy_qdrant_url = "http://dummy_url:6333"
        self.dummy_qdrant_api_key = "dummy_q_key"
        self.dummy_collection_name = "test_collection"
        self.dummy_openai_api_key = "dummy_oai_key"

        # We patch _initialize_vector_store during instantiation for most tests
        # to avoid actual Qdrant client initialization attempts.
        with patch.object(LegalDocumentRAGTool, '_initialize_vector_store', return_value=None) as mock_init_vs:
            self.tool = LegalDocumentRAGTool(
                qdrant_url=self.dummy_qdrant_url,
                qdrant_api_key=self.dummy_qdrant_api_key,
                collection_name=self.dummy_collection_name,
                openai_api_key=self.dummy_openai_api_key
            )
            # Mock the vector_store attribute directly after patched initialization
            self.tool.vector_store = MagicMock()


    def test_rag_tool_run_success(self):
        mock_doc = Document(page_content='Test content from document', metadata={'page': 1})
        self.tool.vector_store.similarity_search.return_value = [mock_doc]

        result = self.tool._run("test query", k=1)

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query", k=1)
        self.assertIn("Retrieved Information from Document:", result)
        self.assertIn("Test content from document", result)
        self.assertIn("Page 1", result)

    def test_rag_tool_run_no_results(self):
        self.tool.vector_store.similarity_search.return_value = []

        result = self.tool._run("test query no results")

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query no results", k=3) # Default k
        self.assertEqual(result, "No relevant information found in the document for your query.")

    def test_rag_tool_initialization_failure_effect(self):
        # Simulate that _initialize_vector_store failed and self.vector_store is None
        with patch.object(LegalDocumentRAGTool, '_initialize_vector_store', return_value=None):
            tool_no_vs = LegalDocumentRAGTool(
                qdrant_url=self.dummy_qdrant_url,
                qdrant_api_key=self.dummy_qdrant_api_key,
                collection_name=self.dummy_collection_name,
                openai_api_key=self.dummy_openai_api_key
            )
            tool_no_vs.vector_store = None # Explicitly set to None

        result = tool_no_vs._run("test query")
        self.assertEqual(result, "Error: RAG Tool's Qdrant vector store is not initialized. Cannot perform search.")

    def test_rag_tool_similarity_search_exception(self):
        self.tool.vector_store.similarity_search.side_effect = Exception("Similarity search failed")

        result = self.tool._run("test query exception")

        self.tool.vector_store.similarity_search.assert_called_once_with(query="test query exception", k=3)
        self.assertEqual(result, "Error during RAG similarity search: Similarity search failed")


@patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.st', mock_st) # Mock streamlit module
class TestProcessUploadedPDFToQdrant(unittest.TestCase):

    def setUp(self):
        self.mock_uploaded_file = MagicMock()
        self.mock_uploaded_file.getvalue.return_value = b"dummy pdf content"
        self.qdrant_url = "http://test-qdrant.com"
        self.qdrant_api_key = "test_q_key"
        self.openai_api_key = "test_oai_key"
        self.collection_name = "test_pdf_collection"

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.os.unlink')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.tempfile.NamedTemporaryFile')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.Qdrant.from_documents')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.OpenAIEmbeddings')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.RecursiveCharacterTextSplitter')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.PyPDFLoader')
    def test_process_pdf_success(self, MockPyPDFLoader, MockTextSplitter, MockOpenAIEmbeddings, MockQdrantFromDocs, MockNamedTempFile, MockOsUnlink):
        # Configure mocks
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "dummy/temp/path.pdf"
        MockNamedTempFile.return_value.__enter__.return_value = mock_tmp_file # For 'with' statement

        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.return_value = [Document(page_content="doc content")]

        mock_splitter_instance = MockTextSplitter.return_value
        mock_texts = [Document(page_content="chunk1")]
        mock_splitter_instance.split_documents.return_value = mock_texts

        mock_embeddings_instance = MockOpenAIEmbeddings.return_value
        mock_qdrant_instance = MagicMock() # Mocked Qdrant vector store instance
        MockQdrantFromDocs.return_value = mock_qdrant_instance

        result = process_uploaded_pdf_to_qdrant(
            self.mock_uploaded_file, self.qdrant_url, self.qdrant_api_key, self.openai_api_key, self.collection_name
        )

        MockNamedTempFile.assert_called_once_with(delete=False, suffix=".pdf")
        mock_tmp_file.write.assert_called_once_with(b"dummy pdf content")
        MockPyPDFLoader.assert_called_once_with(mock_tmp_file.name)
        mock_loader_instance.load.assert_called_once()
        MockTextSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_splitter_instance.split_documents.assert_called_once_with([Document(page_content="doc content")])
        MockOpenAIEmbeddings.assert_called_once_with(api_key=self.openai_api_key)
        MockQdrantFromDocs.assert_called_once_with(
            mock_texts,
            mock_embeddings_instance,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            force_recreate=True
        )
        MockOsUnlink.assert_called_once_with(mock_tmp_file.name)
        self.assertEqual(result, mock_qdrant_instance)


    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.os.unlink')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.tempfile.NamedTemporaryFile')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.PyPDFLoader')
    def test_process_pdf_loader_fails(self, MockPyPDFLoader, MockNamedTempFile, MockOsUnlink):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "dummy/temp/path.pdf"
        MockNamedTempFile.return_value.__enter__.return_value = mock_tmp_file

        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.return_value = [] # Simulate loader failing to load documents

        with patch.object(mock_st, 'error') as mock_st_error_call: # Check st.error
            result = process_uploaded_pdf_to_qdrant(
                self.mock_uploaded_file, self.qdrant_url, self.qdrant_api_key, self.openai_api_key, self.collection_name
            )
            self.assertIsNone(result)
            mock_st_error_call.assert_any_call("Could not load any documents from the PDF.")
        MockOsUnlink.assert_called_once_with(mock_tmp_file.name)


    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.os.unlink')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.tempfile.NamedTemporaryFile')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.RecursiveCharacterTextSplitter')
    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_legal_agent_team.legal_agent_team_crewai.PyPDFLoader')
    def test_process_pdf_splitter_fails(self, MockPyPDFLoader, MockTextSplitter, MockNamedTempFile, MockOsUnlink):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "dummy/temp/path.pdf"
        MockNamedTempFile.return_value.__enter__.return_value = mock_tmp_file

        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.return_value = [Document(page_content="doc content")]

        mock_splitter_instance = MockTextSplitter.return_value
        mock_splitter_instance.split_documents.return_value = [] # Simulate splitter failing

        with patch.object(mock_st, 'error') as mock_st_error_call:
            result = process_uploaded_pdf_to_qdrant(
                self.mock_uploaded_file, self.qdrant_url, self.qdrant_api_key, self.openai_api_key, self.collection_name
            )
            self.assertIsNone(result)
            mock_st_error_call.assert_any_call("Could not split the document into text chunks.")
        MockOsUnlink.assert_called_once_with(mock_tmp_file.name)


if __name__ == "__main__":
    unittest.main()
