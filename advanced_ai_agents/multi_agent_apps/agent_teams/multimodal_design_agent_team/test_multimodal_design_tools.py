import unittest
from unittest.mock import MagicMock, patch
import base64

# Import functions to be tested
from advanced_ai_agents.multi_agent_apps.agent_teams.multimodal_design_agent_team.design_agent_team_crewai import (
    process_uploaded_images,
    format_image_for_gemini
)

# A simple mock for Streamlit's UploadedFile
class MockUploadedFile:
    def __init__(self, name: str, file_type: str, data: bytes):
        self.name = name
        self.type = file_type
        self._data = data

    def getvalue(self) -> bytes:
        if self._data is None: # Simulate an error case
            raise IOError("Failed to get value")
        return self._data

# Mock streamlit module for st.error
mock_st_module = MagicMock()

@patch('advanced_ai_agents.multi_agent_apps.agent_teams.multimodal_design_agent_team.design_agent_team_crewai.st', mock_st_module)
class TestImageProcessingFunctions(unittest.TestCase):

    def test_process_uploaded_images_empty(self):
        result = process_uploaded_images([])
        self.assertEqual(result, [])

    def test_process_uploaded_images_success(self):
        dummy_image_content = b"dummy image data"
        mock_file1 = MockUploadedFile(name="image1.png", file_type="image/png", data=dummy_image_content)
        mock_file2 = MockUploadedFile(name="image2.jpg", file_type="image/jpeg", data=b"other data")

        uploaded_files = [mock_file1, mock_file2]
        processed = process_uploaded_images(uploaded_files)

        self.assertEqual(len(processed), 2)

        # Check first file
        self.assertEqual(processed[0]["name"], "image1.png")
        self.assertEqual(processed[0]["type"], "image/png")
        self.assertEqual(base64.b64decode(processed[0]["base64"]), dummy_image_content)

        # Check second file
        self.assertEqual(processed[1]["name"], "image2.jpg")
        self.assertEqual(processed[1]["type"], "image/jpeg")
        self.assertEqual(base64.b64decode(processed[1]["base64"]), b"other data")

    def test_process_uploaded_images_error_on_one_file(self):
        dummy_image_content = b"valid data"
        mock_file_valid = MockUploadedFile(name="valid.png", file_type="image/png", data=dummy_image_content)
        # This mock file will cause an error because its _data is None, and getvalue will raise IOError
        mock_file_error = MockUploadedFile(name="error.jpg", file_type="image/jpeg", data=None)

        uploaded_files = [mock_file_valid, mock_file_error]

        # Reset mock_st_module.error before the call if it's stateful across tests in a more complex setup
        mock_st_module.error.reset_mock()

        processed = process_uploaded_images(uploaded_files)

        self.assertEqual(len(processed), 1) # Only the valid file should be processed
        self.assertEqual(processed[0]["name"], "valid.png")
        self.assertEqual(base64.b64decode(processed[0]["base64"]), dummy_image_content)

        # Assert that st.error was called due to the error in processing mock_file_error
        mock_st_module.error.assert_called_once()
        # More specific check on the error message if needed:
        # self.assertIn("Error processing image error.jpg", mock_st_module.error.call_args[0][0])


    def test_format_image_for_gemini(self):
        processed_image_data = {
            "name": "test.png",
            "type": "image/png",
            "base64": base64.b64encode(b"dummydata").decode()
        }

        formatted = format_image_for_gemini(processed_image_data)

        expected_url = f"data:{processed_image_data['type']};base64,{processed_image_data['base64']}"
        expected_dict = {
            "type": "image_url",
            "image_url": {"url": expected_url}
        }

        self.assertEqual(formatted, expected_dict)

if __name__ == "__main__":
    unittest.main()
