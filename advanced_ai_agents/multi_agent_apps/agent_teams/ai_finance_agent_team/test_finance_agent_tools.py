import unittest
from unittest.mock import patch, MagicMock
from advanced_ai_agents.multi_agent_apps.agent_teams.ai_finance_agent_team.finance_agent_team_crewai import (
    GetStockPriceTool,
    GetCompanyInfoTool,
    GetCompanyNewsTool,
    GetAnalystRecommendationsTool
)

class TestYahooFinanceTools(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_get_stock_price_tool(self, MockTicker):
        mock_instance = MockTicker.return_value
        mock_instance.info = {"regularMarketPrice": 150.0, "symbol": "DUMMY"}

        tool = GetStockPriceTool()
        result = tool._run("DUMMY_TICKER")

        MockTicker.assert_called_once_with("DUMMY_TICKER")
        self.assertEqual(result, {"regularMarketPrice": 150.0, "symbol": "DUMMY"})

    @patch('yfinance.Ticker')
    def test_get_company_info_tool(self, MockTicker):
        mock_instance = MockTicker.return_value
        mock_instance.info = {"longName": "Dummy Corp", "sector": "Technology"}

        tool = GetCompanyInfoTool()
        result = tool._run("DUMMY_TICKER")

        MockTicker.assert_called_once_with("DUMMY_TICKER")
        self.assertEqual(result, {"longName": "Dummy Corp", "sector": "Technology"})

    @patch('yfinance.Ticker')
    def test_get_company_news_tool(self, MockTicker):
        mock_instance = MockTicker.return_value
        mock_instance.news = [{"title": "Dummy News 1", "link": "http://dummy.com/news1"}]

        tool = GetCompanyNewsTool()
        result = tool._run("DUMMY_TICKER")

        MockTicker.assert_called_once_with("DUMMY_TICKER")
        self.assertEqual(result, [{"title": "Dummy News 1", "link": "http://dummy.com/news1"}])

    @patch('yfinance.Ticker')
    def test_get_analyst_recommendations_tool(self, MockTicker):
        mock_instance = MockTicker.return_value
        mock_instance.recommendations = [{"firm": "Analyst Firm A", "toGrade": "Buy"}]

        tool = GetAnalystRecommendationsTool()
        result = tool._run("DUMMY_TICKER")

        MockTicker.assert_called_once_with("DUMMY_TICKER")
        self.assertEqual(result, [{"firm": "Analyst Firm A", "toGrade": "Buy"}])

if __name__ == "__main__":
    unittest.main()
