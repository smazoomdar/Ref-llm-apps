import unittest
from unittest.mock import patch, MagicMock

# Import the tool to be tested
from advanced_ai_agents.multi_agent_apps.agent_teams.multimodal_coding_agent_team.ai_coding_agent_o3_crewai import (
    E2BCodeExecutionTool
)

# Import E2BSandbox for type hinting and creating mock specs if needed
# In the main file, it's imported as: from e2b import Sandbox as E2BSandbox
from e2b import Sandbox as E2BSandboxType # Use a different alias to avoid confusion if any

class TestE2BCodeExecutionTool(unittest.TestCase):

    def setUp(self):
        # Create a generic mock sandbox instance for most tests
        self.mock_sandbox_instance = MagicMock(spec=E2BSandboxType)

    def test_execution_success(self):
        # Configure the mock sandbox's exec_cell behavior for this test
        mock_execution_result = MagicMock()
        mock_execution_result.outputs = [MagicMock(name='stdout', text='Hello World')]
        mock_execution_result.error = None
        self.mock_sandbox_instance.notebook.exec_cell.return_value = mock_execution_result

        tool = E2BCodeExecutionTool(sandbox_instance=self.mock_sandbox_instance)
        code_to_run = "print('Hello World')"
        result = tool._run(code_to_run)

        self.mock_sandbox_instance.notebook.exec_cell.assert_called_once_with(code_to_run)
        self.assertEqual(result["stdout"], "Hello World")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["error"])
        self.assertIn("execution_outputs", result)

    def test_execution_with_stderr(self):
        mock_execution_result = MagicMock()
        mock_execution_result.outputs = [MagicMock(name='stderr', text='Warning!')]
        mock_execution_result.error = None
        self.mock_sandbox_instance.notebook.exec_cell.return_value = mock_execution_result

        tool = E2BCodeExecutionTool(sandbox_instance=self.mock_sandbox_instance)
        code_to_run = "import sys; sys.stderr.write('Warning!')"
        result = tool._run(code_to_run)

        self.mock_sandbox_instance.notebook.exec_cell.assert_called_once_with(code_to_run)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "Warning!")
        self.assertIsNone(result["error"])

    def test_execution_with_runtime_error(self):
        mock_execution_result = MagicMock()
        mock_execution_result.outputs = []
        mock_execution_result.error = MagicMock(name='ValueError', value='Some error', traceback='Traceback info...')
        self.mock_sandbox_instance.notebook.exec_cell.return_value = mock_execution_result

        tool = E2BCodeExecutionTool(sandbox_instance=self.mock_sandbox_instance)
        code_to_run = "raise ValueError('Some error')"
        result = tool._run(code_to_run)

        self.mock_sandbox_instance.notebook.exec_cell.assert_called_once_with(code_to_run)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["error"], "ValueError")

    def test_sandbox_not_available(self):
        # Instantiate tool ensuring no sandbox instance and no API key to init one
        tool = E2BCodeExecutionTool(sandbox_instance=None, e2b_api_key=None)

        result = tool._run("print('test')")

        self.assertEqual(result["error"], "E2B Sandbox is not available to the tool.")
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")

    @patch('advanced_ai_agents.multi_agent_apps.agent_teams.multimodal_coding_agent_team.ai_coding_agent_o3_crewai.E2BSandbox')
    def test_tool_initializes_own_sandbox_if_key_provided(self, MockE2BSandboxInToolFile):
        # This tests the fallback in the tool's __init__
        mock_internal_sandbox = MockE2BSandboxInToolFile.return_value # The instance created by the tool

        # Mock the exec_cell method on this internally created instance
        mock_execution_result = MagicMock()
        mock_execution_result.outputs = [MagicMock(name='stdout', text='Internal Sandbox Test')]
        mock_execution_result.error = None
        mock_internal_sandbox.notebook.exec_cell.return_value = mock_execution_result

        dummy_api_key = "test_e2b_api_key"
        tool = E2BCodeExecutionTool(sandbox_instance=None, e2b_api_key=dummy_api_key)

        # Check if E2BSandbox (as imported in the tool's file) was called during __init__
        MockE2BSandboxInToolFile.assert_called_once_with(api_key=dummy_api_key)

        # Now run the tool to see if it uses the internally created sandbox
        code_to_run = "print('Internal Sandbox Test')"
        result = tool._run(code_to_run)

        self.assertIsNotNone(tool.sandbox, "Tool should have an initialized sandbox instance.")
        mock_internal_sandbox.notebook.exec_cell.assert_called_once_with(code_to_run)
        self.assertEqual(result["stdout"], "Internal Sandbox Test")
        self.assertIsNone(result["error"])

    def test_execution_sandbox_exception(self):
        # Test scenario where sandbox.notebook.exec_cell itself raises an exception
        self.mock_sandbox_instance.notebook.exec_cell.side_effect = Exception("E2B SDK Exception")

        tool = E2BCodeExecutionTool(sandbox_instance=self.mock_sandbox_instance)
        code_to_run = "some code"
        result = tool._run(code_to_run)

        self.mock_sandbox_instance.notebook.exec_cell.assert_called_once_with(code_to_run)
        self.assertIn("Error during code execution in E2B: E2B SDK Exception", result["error"])
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")


if __name__ == "__main__":
    unittest.main()
