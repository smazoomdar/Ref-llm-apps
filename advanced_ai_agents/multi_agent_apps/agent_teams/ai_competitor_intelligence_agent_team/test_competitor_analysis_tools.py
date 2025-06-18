import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from typing import List

# Import necessary classes from the main script
from advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai import (
    FirecrawlSchemaExtractionTool,
    get_competitor_urls,
    CompetitorDataSchema # Assuming this schema is used or a similar one for testing
)

# A dummy Pydantic model for testing schema extraction
class TestSchema(BaseModel):
    name: str = Field(...)
    value: int = Field(...)

class TestCompetitorAnalysisTools(unittest.TestCase):

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai.FirecrawlApp')
    def test_firecrawl_schema_extraction_tool_success(self, MockFirecrawlApp):
        # Setup mock
        mock_firecrawl_instance = MockFirecrawlApp.return_value
        mock_firecrawl_instance.extract.return_value = {"data": {"name": "TestCo", "value": 123}} # Adjusted to match typical Firecrawl output

        tool = FirecrawlSchemaExtractionTool()
        result = tool._run(url="http://dummy.com", schema=TestSchema, firecrawl_api_key="dummy_key")

        MockFirecrawlApp.assert_called_once_with(api_key="dummy_key")
        mock_firecrawl_instance.extract.assert_called_once_with(
            url="http://dummy.com",
            extraction_schema=TestSchema.model_json_schema()
        )
        self.assertEqual(result, {"data": {"name": "TestCo", "value": 123}})

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai.FirecrawlApp')
    def test_firecrawl_schema_extraction_tool_error(self, MockFirecrawlApp):
        # Setup mock to raise an exception
        mock_firecrawl_instance = MockFirecrawlApp.return_value
        mock_firecrawl_instance.extract.side_effect = Exception("Firecrawl API error")

        tool = FirecrawlSchemaExtractionTool()
        result = tool._run(url="http://error.com", schema=TestSchema, firecrawl_api_key="dummy_key")

        MockFirecrawlApp.assert_called_once_with(api_key="dummy_key")
        self.assertIn("error", result)
        self.assertTrue("Failed to extract data from http://error.com: Firecrawl API error" in result["error"])

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai.Exa')
    def test_get_competitor_urls_success(self, MockExa):
        # Setup mock Exa response
        mock_exa_instance = MockExa.return_value
        mock_search_result_item = MagicMock()
        mock_search_result_item.url = "http://competitor1.com"

        mock_search_result_item2 = MagicMock()
        mock_search_result_item2.url = "http://competitor2.com"

        mock_search_result_item_dup = MagicMock()
        mock_search_result_item_dup.url = "http://competitor1.com" # Duplicate

        mock_response = MagicMock()
        mock_response.results = [mock_search_result_item, mock_search_result_item2, mock_search_result_item_dup]
        mock_exa_instance.search_and_contents.return_value = mock_response

        urls = get_competitor_urls("http://mycompany.com", "My company description", "exa_dummy_key")

        MockExa.assert_called_once_with(api_key="exa_dummy_key")
        mock_exa_instance.search_and_contents.assert_called_once()
        self.assertEqual(urls, ["http://competitor1.com", "http://competitor2.com"])

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai.Exa')
    def test_get_competitor_urls_no_results(self, MockExa):
        mock_exa_instance = MockExa.return_value
        mock_response = MagicMock()
        mock_response.results = []
        mock_exa_instance.search_and_contents.return_value = mock_response

        urls = get_competitor_urls("http://mycompany.com", "My company", "exa_dummy_key")
        self.assertEqual(urls, [])

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.ai_competitor_intelligence_agent_team.competitor_agent_team_crewai.Exa')
    def test_get_competitor_urls_exa_exception(self, MockExa):
        mock_exa_instance = MockExa.return_value
        mock_exa_instance.search_and_contents.side_effect = Exception("Exa API error")

        # Streamlit's st.error will be called, we can check for that if we mock st
        # For now, just ensure it returns an empty list and doesn't crash
        with patch('streamlit.error') as mock_st_error: # Mock streamlit's error function
            urls = get_competitor_urls("http://mycompany.com", "My company", "exa_dummy_key")
            self.assertEqual(urls, [])
            mock_st_error.assert_called_once() # Check that st.error was called

if __name__ == "__main__":
    unittest.main()
