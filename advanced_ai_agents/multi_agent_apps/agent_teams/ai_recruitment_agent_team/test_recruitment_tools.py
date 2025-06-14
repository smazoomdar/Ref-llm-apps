import unittest
from unittest.mock import patch, MagicMock
import smtplib
import requests
from datetime import datetime, timedelta

# Import tools to be tested
from advanced_ai_agents.multi_agent_apps.agent_teams.ai_recruitment_agent_team.ai_recruitment_agent_crewai import (
    SendEmailTool,
    ScheduleZoomMeetingTool
)

class TestSendEmailTool(unittest.TestCase):

    def setUp(self):
        self.sender_email = "test_sender@example.com"
        self.sender_password = "dummy_password"
        self.smtp_server = "smtp.example.com"
        self.smtp_port = 587
        self.tool = SendEmailTool(
            sender_email=self.sender_email,
            sender_password=self.sender_password,
            smtp_server=self.smtp_server,
            smtp_port=self.smtp_port
        )

    @patch('smtplib.SMTP')
    def test_send_email_success(self, MockSMTP):
        mock_server = MockSMTP.return_value.__enter__.return_value # For 'with' statement

        recipient = "recipient@example.com"
        subject = "Test Subject"
        body = "Test Body"

        result = self.tool._run(recipient, subject, body)

        MockSMTP.assert_called_once_with(self.smtp_server, self.smtp_port)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with(self.sender_email, self.sender_password)
        mock_server.sendmail.assert_called_once()
        # Check that the first argument to sendmail is the sender email
        self.assertEqual(mock_server.sendmail.call_args[0][0], self.sender_email)
        # Check that the second argument is the recipient
        self.assertEqual(mock_server.sendmail.call_args[0][1], recipient)
        # Check that the message body contains subject and body (MIMEText structure)
        sent_message = mock_server.sendmail.call_args[0][2]
        self.assertIn(f"Subject: {subject}", sent_message)
        self.assertIn(body, sent_message)

        self.assertEqual(result, f"Success: Email sent to {recipient} with subject '{subject}'.")

    @patch('smtplib.SMTP')
    def test_send_email_failure(self, MockSMTP):
        mock_server = MockSMTP.return_value.__enter__.return_value
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid")

        result = self.tool._run("recipient@example.com", "Test Subject Fail", "Test Body Fail")

        self.assertIn("Error sending email: (535, b'Authentication credentials invalid')", result)

    def test_send_email_missing_config(self):
        tool_missing_config = SendEmailTool(
            sender_email=None, # Missing sender_email
            sender_password=self.sender_password,
            smtp_server=self.smtp_server,
            smtp_port=self.smtp_port
        )
        result = tool_missing_config._run("recipient@example.com", "Test Subject", "Test Body")
        self.assertEqual(result, "Error: Email sender credentials or server configuration is missing.")


class TestScheduleZoomMeetingTool(unittest.TestCase):

    def setUp(self):
        self.zoom_account_id = "dummy_account_id"
        self.zoom_client_id = "dummy_client_id"
        self.zoom_client_secret = "dummy_secret"
        self.tool = ScheduleZoomMeetingTool(
            zoom_account_id=self.zoom_account_id,
            zoom_client_id=self.zoom_client_id,
            zoom_client_secret=self.zoom_client_secret
        )
        # Reset internal token cache for each test
        self.tool._access_token = None
        self.tool._token_expires_at = None


    @patch('requests.post')
    def test_get_access_token_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "new_token", "expires_in": 3600}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # First call - should fetch token
        token = self.tool._get_access_token()
        self.assertEqual(token, "new_token")
        mock_post.assert_called_once()
        self.assertIsNotNone(self.tool._access_token)
        self.assertIsNotNone(self.tool._token_expires_at)

        # Second call - should use cached token
        cached_token = self.tool._get_access_token()
        self.assertEqual(cached_token, "new_token")
        mock_post.assert_called_once() # Still called only once

    @patch('requests.post')
    def test_get_access_token_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("API error")

        token = self.tool._get_access_token()
        self.assertIsNone(token)
        self.assertIsNone(self.tool._access_token) # Should not be set

    @patch('requests.post') # This will mock requests.post for meeting creation
    def test_run_schedule_meeting_success(self, mock_meeting_post):
        # Mock _get_access_token to avoid actual auth call in this specific test
        with patch.object(self.tool, '_get_access_token', return_value="dummy_access_token"):
            mock_meeting_response = MagicMock()
            mock_meeting_response.json.return_value = {"id": "12345", "join_url": "http://zoom.us/join/12345"}
            mock_meeting_response.raise_for_status.return_value = None
            mock_meeting_post.return_value = mock_meeting_response

            topic = "Test Meeting Success"
            start_time = "2024-01-01T10:00:00Z"
            duration = 60
            attendees = ["attendee@example.com"]

            result = self.tool._run(topic, start_time, duration, attendees)

            self.tool._get_access_token.assert_called_once()
            mock_meeting_post.assert_called_once()
            args, kwargs = mock_meeting_post.call_args
            self.assertEqual(args[0], "https://api.zoom.us/v2/users/me/meetings")
            self.assertEqual(kwargs['headers']['Authorization'], "Bearer dummy_access_token")
            self.assertEqual(kwargs['json']['topic'], topic)

            self.assertIn("Success: Zoom meeting scheduled. ID: 12345, Join URL: http://zoom.us/join/12345", result)

    def test_run_schedule_meeting_token_failure(self):
        with patch.object(self.tool, '_get_access_token', return_value=None):
            result = self.tool._run("Test Meeting Token Fail", "2024-01-01T11:00:00Z", 30, [])
            self.tool._get_access_token.assert_called_once()
            self.assertEqual(result, "Error: Could not obtain Zoom access token.")

    @patch('requests.post') # For meeting creation
    def test_run_schedule_meeting_api_failure(self, mock_meeting_post):
        with patch.object(self.tool, '_get_access_token', return_value="dummy_access_token"):
            mock_error_response = MagicMock()
            mock_error_response.json.return_value = {"code": 300, "message": "Invalid meeting params"}
            # Simulate an HTTPError by raising it from the mock
            http_error = requests.exceptions.HTTPError(response=mock_error_response)
            http_error.response.raise_for_status = MagicMock(side_effect=http_error) # make raise_for_status raise itself
            mock_meeting_post.side_effect = http_error # Raise the error when requests.post is called

            result = self.tool._run("Test Meeting API Fail", "2024-01-01T12:00:00Z", 45, [])

            self.tool._get_access_token.assert_called_once()
            mock_meeting_post.assert_called_once() # Ensure it was called
            self.assertIn("Error scheduling Zoom meeting: {'code': 300, 'message': 'Invalid meeting params'}", result)

    def test_run_missing_zoom_credentials(self):
        tool_no_creds = ScheduleZoomMeetingTool(
            zoom_account_id=None, # Missing account_id
            zoom_client_id=self.zoom_client_id,
            zoom_client_secret=self.zoom_client_secret
        )
        result = tool_no_creds._run("Test Meeting No Creds", "2024-01-01T13:00:00Z", 60, [])
        self.assertEqual(result, "Error: Zoom API credentials (Account ID, Client ID, Client Secret) are missing.")


if __name__ == "__main__":
    unittest.main()
