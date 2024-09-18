import unittest
import torch
from demo_chatbot import construct_history, StopOnTokens


class ChatbotTestCase(unittest.TestCase):

    def test_message_history(self):
        # Define a sample message and conversation history
        message = "What's the weather today?"
        history = [["Hey!", "Hi, how can I help you?"]]
        # Run the predict function (it's a generator, so we'll need to iterate over it)
        constructed_history = construct_history(message, history)
        # Expected format in the joined history
        expected_result = "\n<human>:Hey!\n<bot>:Hi, how can I help you?\n<human>:What's the weather today?\n<bot>:"
        # Assert that the result starts with the expected formatted history
        self.assertEqual(expected_result, constructed_history, msg="\n\nError: The message history is not correctly being constructed. Revise the code.")

    def test_StopToken(self):
        stop_criteria = StopOnTokens()
        input_ids_with_stop_token = torch.tensor([[1, 2, 3, 29]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        self.assertTrue(stop_criteria(input_ids_with_stop_token, scores), msg="\n\nError: Something with the Stop Token is not working properly.")

if __name__ == '__main__':
    unittest.main()