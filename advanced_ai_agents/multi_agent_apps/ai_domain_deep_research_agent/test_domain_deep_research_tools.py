import unittest
from unittest.mock import patch, MagicMock
import json # For formatting raw responses in messages

# Import the tool to be tested
from advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai import (
    ComposioBasedGoogleDocTool
)

# Import Composio specific classes for mocking and type hinting if needed
from composio.phidata.toolset import ComposioToolSet
from composio.phidata.action import Action
from composio.client.collections import App # For App.GOOGLEDOCS

class TestComposioBasedGoogleDocTool(unittest.TestCase):

    def setUp(self):
        self.dummy_composio_api_key = "test_composio_key"

    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_tool_initialization_success(self, MockComposioToolSet):
        mock_toolset_instance = MockComposioToolSet.return_value

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)

        MockComposioToolSet.assert_called_once_with(api_key=self.dummy_composio_api_key)
        mock_toolset_instance.enable_app.assert_called_once_with(App.GOOGLEDOCS)
        self.assertIsNotNone(tool.toolset)

    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet', side_effect=Exception("Composio Init Failed"))
    @patch('builtins.print') # To capture warning print
    def test_tool_initialization_failure(self, mock_print, MockComposioToolSet):
        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)

        MockComposioToolSet.assert_called_once_with(api_key=self.dummy_composio_api_key)
        self.assertIsNone(tool.toolset) # Toolset should be None if init fails
        mock_print.assert_any_call("Warning: ComposioBasedGoogleDocTool failed to initialize ComposioToolSet or enable GoogleDocs during __init__: Composio Init Failed")

        # Test the retry logic in _run
        # Now, when _run is called, it will try to initialize ComposioToolSet again,
        # and since it's still patched to fail, it should return the initialization error.
        result = tool._run("Test Doc", "Test Content")
        self.assertIn("Error: ComposioToolSet could not be initialized: Composio Init Failed", result)


    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_run_success_with_direct_link(self, MockComposioToolSet):
        mock_toolset_instance = MockComposioToolSet.return_value
        mock_gdocs_action = MagicMock()
        mock_toolset_instance.get_action.return_value = mock_gdocs_action
        mock_gdocs_action.execute.return_value = {"document_url": "http://direct.link"}

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)
        result = tool._run("Test Doc Direct", "Test Content")

        mock_toolset_instance.get_action.assert_called_once_with(Action.GOOGLEDOCS_CREATE_DOCUMENT)
        mock_gdocs_action.execute.assert_called_once_with(params={"name": "Test Doc Direct", "content": "Test Content"})
        self.assertEqual(result, "Success: Google Doc 'Test Doc Direct' created. Link: http://direct.link")

    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_run_success_with_webview_link(self, MockComposioToolSet): # Test webViewLink as another option
        mock_toolset_instance = MockComposioToolSet.return_value
        mock_gdocs_action = MagicMock()
        mock_toolset_instance.get_action.return_value = mock_gdocs_action
        mock_gdocs_action.execute.return_value = {"webViewLink": "http://webview.link"}

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)
        result = tool._run("Test Doc WebView", "Test Content")
        self.assertEqual(result, "Success: Google Doc 'Test Doc WebView' created. Link: http://webview.link")


    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_run_success_with_constructed_link(self, MockComposioToolSet):
        mock_toolset_instance = MockComposioToolSet.return_value
        mock_gdocs_action = MagicMock()
        mock_toolset_instance.get_action.return_value = mock_gdocs_action
        mock_response_data = {"documentId": "doc_id_123", "name": "Test Doc ID"}
        mock_gdocs_action.execute.return_value = mock_response_data

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)
        result = tool._run("Test Doc ID", "Test Content")

        expected_link = "https://docs.google.com/document/d/doc_id_123/edit"
        self.assertEqual(result, f"Success: Google Doc 'Test Doc ID' created. Link (constructed): {expected_link}. Raw response: {json.dumps(mock_response_data)}")

    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_run_success_no_link_or_id_in_response(self, MockComposioToolSet):
        mock_toolset_instance = MockComposioToolSet.return_value
        mock_gdocs_action = MagicMock()
        mock_toolset_instance.get_action.return_value = mock_gdocs_action
        mock_response_data = {"some_other_field": "value", "name": "Test Doc No Link"}
        mock_gdocs_action.execute.return_value = mock_response_data

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)
        result = tool._run("Test Doc No Link", "Test Content")

        self.assertEqual(result, f"Warning: Google Doc 'Test Doc No Link' created, but link not found in response. Response: {json.dumps(mock_response_data)}")

    @patch('advanced_ai_agents.multi_agent_apps.ai_domain_deep_research_agent.ai_domain_deep_research_agent_crewai.ComposioToolSet')
    def test_run_composio_execute_raises_exception(self, MockComposioToolSet):
        mock_toolset_instance = MockComposioToolSet.return_value
        mock_gdocs_action = MagicMock()
        mock_toolset_instance.get_action.return_value = mock_gdocs_action
        mock_gdocs_action.execute.side_effect = Exception("Composio Execute Failed")

        tool = ComposioBasedGoogleDocTool(composio_api_key=self.dummy_composio_api_key)
        result = tool._run("Test Doc Exception", "Test Content")

        self.assertEqual(result, "Error using ComposioBasedGoogleDocTool: Composio Execute Failed. Ensure Composio Google Docs app is connected and permissions are correct.")

    def test_run_missing_composio_key_at_init(self):
        with self.assertRaises(ValueError) as context:
            ComposioBasedGoogleDocTool(composio_api_key=None)
        self.assertIn("Composio API key is required for ComposioBasedGoogleDocTool.", str(context.exception))

if __name__ == "__main__":
    unittest.main()
