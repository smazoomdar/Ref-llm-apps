import unittest
from unittest.mock import patch, MagicMock
from googleapiclient.errors import HttpError

# Assuming app_crewai.py is in the same directory or accessible in PYTHONPATH
from mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai import CreateCalendarEventTool

class TestCreateCalendarEventTool(unittest.TestCase):

    def setUp(self):
        self.dummy_client_id = "dummy_id"
        self.dummy_client_secret = "dummy_secret"
        self.dummy_refresh_token = "dummy_token"
        self.tool = CreateCalendarEventTool(
            google_client_id=self.dummy_client_id,
            google_client_secret=self.dummy_client_secret,
            google_refresh_token=self.dummy_refresh_token
        )
        self.sample_event_data = {
            "summary": "Test Event",
            "start_datetime_str": "2024-01-01T10:00:00Z",
            "end_datetime_str": "2024-01-01T11:00:00Z",
            "description": "A test event.",
            "location": "Test Location",
            "timezone": "UTC"
        }

    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.build')
    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.Credentials')
    def test_successful_event_creation(self, MockCredentials, mock_build):
        # Mock Credentials
        mock_creds_instance = MockCredentials.from_authorized_user_info.return_value
        mock_creds_instance.valid = True

        # Mock Google API Service and Event Creation
        mock_service_instance = mock_build.return_value
        mock_events_resource = mock_service_instance.events.return_value
        mock_insert_method = mock_events_resource.insert.return_value
        mock_insert_method.execute.return_value = {'htmlLink': 'http://some.link', 'summary': 'Test Event'}

        result = self.tool._run(**self.sample_event_data)

        MockCredentials.from_authorized_user_info.assert_called_once_with(
            info={
                "client_id": self.dummy_client_id,
                "client_secret": self.dummy_client_secret,
                "refresh_token": self.dummy_refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token",
            },
            scopes=["https://www.googleapis.com/auth/calendar.events"]
        )
        mock_build.assert_called_once_with("calendar", "v3", credentials=mock_creds_instance)
        mock_events_resource.insert.assert_called_once_with(
            calendarId="primary",
            body={
                "summary": self.sample_event_data["summary"],
                "location": self.sample_event_data["location"],
                "description": self.sample_event_data["description"],
                "start": {"dateTime": self.sample_event_data["start_datetime_str"], "timeZone": self.sample_event_data["timezone"]},
                "end": {"dateTime": self.sample_event_data["end_datetime_str"], "timeZone": self.sample_event_data["timezone"]},
            }
        )
        mock_insert_method.execute.assert_called_once()
        self.assertIn("Success: Event created", result)
        self.assertIn("http://some.link", result)

    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.Request') # Mock google.auth.transport.requests.Request
    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.Credentials')
    def test_token_refresh_failure(self, MockCredentials, MockRequest):
        mock_creds_instance = MockCredentials.from_authorized_user_info.return_value
        mock_creds_instance.valid = False
        mock_creds_instance.expired = True
        mock_creds_instance.refresh_token = self.dummy_refresh_token # Ensure refresh token exists
        mock_creds_instance.refresh.side_effect = Exception("Refresh failed")

        result = self.tool._run(**self.sample_event_data)

        mock_creds_instance.refresh.assert_called_once_with(MockRequest())
        self.assertIn("Error refreshing Google Calendar token: Refresh failed", result)

    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.build')
    @patch('mcp_ai_agents.ai_travel_planner_mcp_agent_team.app_crewai.Credentials')
    def test_google_api_http_error(self, MockCredentials, mock_build):
        mock_creds_instance = MockCredentials.from_authorized_user_info.return_value
        mock_creds_instance.valid = True

        mock_service_instance = mock_build.return_value
        mock_events_resource = mock_service_instance.events.return_value

        # Simulate HttpError
        # The HttpError constructor typically takes resp (a Response object) and content (bytes)
        mock_http_error_response = MagicMock()
        mock_http_error_response.status = 403
        # HttpError expects content to be bytes
        http_error_instance = HttpError(resp=mock_http_error_response, content=b'Forbidden by API rules')

        mock_insert_method = mock_events_resource.insert.return_value
        mock_insert_method.execute.side_effect = http_error_instance

        result = self.tool._run(**self.sample_event_data)

        self.assertIn("Error creating Google Calendar event", result)
        # The string representation of HttpError includes details like status and reason
        self.assertIn("Forbidden by API rules", str(result)) # Check if the error message content is in the result

    def test_missing_credentials_at_initialization(self):
        # Test if tool handles missing credentials if they were not passed at init
        # This specific tool expects them at init, so this tests the _run time check (though init is more critical)
        tool_no_creds_at_run = CreateCalendarEventTool(google_client_id=None, google_client_secret=None, google_refresh_token=None)
        result = tool_no_creds_at_run._run(**self.sample_event_data)
        self.assertIn("Error: Google API credentials (Client ID, Client Secret, Refresh Token) are missing", result)

if __name__ == "__main__":
    unittest.main()
