import unittest
from unittest.mock import patch, MagicMock
import os

# Import the class to be tested and its dependent schema
from advanced_ai_agents.multi_agent_apps.ai_aqi_analysis_agent.ai_aqi_analysis_agent_gradio_crewai import (
    AQIAnalyzer,
    ExtractSchema # Though not directly used in tests, good to ensure it's importable
)

class TestAQIAnalyzer(unittest.TestCase):

    def setUp(self):
        self.dummy_firecrawl_api_key = "test_firecrawl_key"
        # Patch os.getenv for FIRECRAWL_API_KEY to ensure tests are isolated
        # from actual environment variables during AQIAnalyzer instantiation if key not passed.
        self.env_patch = patch.dict(os.environ, {"FIRECRAWL_API_KEY": self.dummy_firecrawl_api_key})
        self.env_patch.start()
        self.analyzer = AQIAnalyzer(firecrawl_api_key=self.dummy_firecrawl_api_key)

    def tearDown(self):
        self.env_patch.stop()

    @patch.object(AQIAnalyzer, 'app', new_callable=MagicMock) # Mock the FirecrawlApp instance within AQIAnalyzer
    def test_fetch_aqi_data_success(self, mock_firecrawl_app_instance):
        city, state, country = "TestCity", "TestState", "TestCountry"
        mock_search_results = [
            {"url": "http://someotheraqisite.com/testcity"},
            {"url": "http://aqicn.org/city/testcity"}, # This one should be picked
            {"url": "http://anothersite.com"}
        ]
        mock_firecrawl_app_instance.search.return_value = mock_search_results

        mock_extracted_llm_data = {
            "aqi_value": 50,
            "main_pollutant": "PM2.5",
            "health_implications": "Moderate health concern.",
            "cautionary_statements": ["Sensitive groups should reduce outdoor activity."]
        }
        mock_scrape_return_value = {"llm_extraction": mock_extracted_llm_data}
        mock_firecrawl_app_instance.scrape_page.return_value = mock_scrape_return_value

        result = self.analyzer.fetch_aqi_data(city, state, country)

        self.assertEqual(result, mock_extracted_llm_data)
        mock_firecrawl_app_instance.search.assert_called_once_with(
            f"Current AQI data for {city}, {state}, {country}",
            page_options={"fetch_page_content": False, "limit": 3}
        )
        mock_firecrawl_app_instance.scrape_page.assert_called_once_with(
            "http://aqicn.org/city/testcity", # Assert the correct URL was chosen
            params={
                "pageOptions": {"onlyMainContent": True},
                "extractorOptions": {
                    "mode": "llm-extraction",
                    "extractionPrompt": f"Extract the AQI value, main pollutant, health implications, and cautionary statements for {city}, {state}, {country}. If multiple AQI values are present (e.g., US AQI, CN AQI), prioritize US AQI or a general one if US AQI is not available.",
                    "extractionSchema": ExtractSchema.model_json_schema()
                }
            }
        )

    @patch.object(AQIAnalyzer, 'app', new_callable=MagicMock)
    def test_fetch_aqi_data_no_search_results(self, mock_firecrawl_app_instance):
        mock_firecrawl_app_instance.search.return_value = [] # No search results

        result = self.analyzer.fetch_aqi_data("AnyCity", "AnyState", "AnyCountry")

        self.assertIsNone(result)
        mock_firecrawl_app_instance.search.assert_called_once()
        mock_firecrawl_app_instance.scrape_page.assert_not_called() # Scrape should not be called

    @patch.object(AQIAnalyzer, 'app', new_callable=MagicMock)
    def test_fetch_aqi_data_scrape_fails(self, mock_firecrawl_app_instance):
        mock_search_results = [{"url": "http://aqicn.org/city/testcity"}]
        mock_firecrawl_app_instance.search.return_value = mock_search_results
        mock_firecrawl_app_instance.scrape_page.side_effect = Exception("Scraping error") # Simulate scrape failure

        result = self.analyzer.fetch_aqi_data("TestCity", "TestState", "TestCountry")

        self.assertIsNone(result)
        mock_firecrawl_app_instance.search.assert_called_once()
        mock_firecrawl_app_instance.scrape_page.assert_called_once() # It was called but failed

    @patch.object(AQIAnalyzer, 'app', new_callable=MagicMock)
    def test_fetch_aqi_data_no_llm_extraction(self, mock_firecrawl_app_instance):
        mock_search_results = [{"url": "http://aqicn.org/city/testcity"}]
        mock_firecrawl_app_instance.search.return_value = mock_search_results
        mock_firecrawl_app_instance.scrape_page.return_value = {"llm_extraction": None} # No LLM data

        result = self.analyzer.fetch_aqi_data("TestCity", "TestState", "TestCountry")

        self.assertIsNone(result) # Should return None if llm_extraction is empty or None
        mock_firecrawl_app_instance.search.assert_called_once()
        mock_firecrawl_app_instance.scrape_page.assert_called_once()

    @patch.dict(os.environ, clear=True) # Clear all env vars for this test
    def test_fetch_aqi_data_firecrawl_api_key_missing_at_init(self):
        # Unset FIRECRAWL_API_KEY for the scope of this test if AQIAnalyzer tries to get it from env
        # The patch.dict above already clears os.environ for this test method.
        # If os.getenv("FIRECRAWL_API_KEY") is called inside __init__ and returns None,
        # and no key is passed to constructor, it should raise ValueError.

        with self.assertRaises(ValueError) as context:
            AQIAnalyzer(firecrawl_api_key=None)
        self.assertIn("Firecrawl API Key is required.", str(context.exception))

    @patch.object(AQIAnalyzer, 'app', new_callable=MagicMock)
    def test_fetch_aqi_data_uses_first_search_result_as_fallback(self, mock_firecrawl_app_instance):
        city, state, country = "FallbackCity", "FallbackState", "FallbackCountry"
        # No preferred URLs, only other URLs
        mock_search_results = [
            {"url": "http://fallbacksite1.com/fallbackcity"},
            {"url": "http://fallbacksite2.com"}
        ]
        mock_firecrawl_app_instance.search.return_value = mock_search_results

        mock_extracted_llm_data = {"aqi_value": 75, "main_pollutant": "O3"} # Simplified
        mock_scrape_return_value = {"llm_extraction": mock_extracted_llm_data}
        mock_firecrawl_app_instance.scrape_page.return_value = mock_scrape_return_value

        result = self.analyzer.fetch_aqi_data(city, state, country)

        self.assertEqual(result, mock_extracted_llm_data)
        mock_firecrawl_app_instance.scrape_page.assert_called_once()
        # Check that the first URL from search_results was used for scraping
        self.assertEqual(mock_firecrawl_app_instance.scrape_page.call_args[0][0], "http://fallbacksite1.com/fallbackcity")


if __name__ == "__main__":
    unittest.main()
