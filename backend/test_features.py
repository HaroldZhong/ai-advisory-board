import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import calculate_cost
from backend.analytics import get_analytics
from backend.logger import logger

class TestFeatures(unittest.TestCase):
    def test_calculate_cost(self):
        # Mock usage
        usage = {'prompt_tokens': 1000, 'completion_tokens': 1000}
        
        # Test with GPT-4o Mini (Input: 0.15, Output: 0.6)
        # Cost = (1000/1M * 0.15) + (1000/1M * 0.6) = 0.00015 + 0.0006 = 0.00075
        cost = calculate_cost(usage, "openai/gpt-4o-mini")
        self.assertAlmostEqual(cost, 0.00075)
        
        # Test with unknown model
        cost = calculate_cost(usage, "unknown-model")
        self.assertEqual(cost, 0.0)
        
        # Test with empty usage
        cost = calculate_cost({}, "openai/gpt-4o-mini")
        self.assertEqual(cost, 0.0)

    @patch('backend.analytics.os.listdir')
    @patch('backend.analytics.open')
    @patch('backend.analytics.os.path.exists')
    def test_analytics(self, mock_exists, mock_open, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ['test_conv.json']
        
        # Mock conversation data
        mock_data = {
            "messages": [
                {
                    "stage2": [
                        {
                            "model": "judge1",
                            "parsed_ranking": ["Response A", "Response B"]
                        }
                    ],
                    "metadata": {
                        "label_to_model": {
                            "Response A": "model_a",
                            "Response B": "model_b"
                        }
                    }
                }
            ]
        }
        
        # Setup mock file read
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(mock_data)
        mock_open.return_value = mock_file
        
        # Run analytics
        results = get_analytics()
        
        # Verify results
        models = results['models']
        self.assertEqual(len(models), 2)
        
        # Model A should be rank 1 (avg 1.0), win rate 100%
        model_a = next(m for m in models if m['model'] == 'model_a')
        self.assertEqual(model_a['average_rank'], 1.0)
        self.assertEqual(model_a['win_rate'], 100.0)
        
        # Model B should be rank 2 (avg 2.0), win rate 0%
        model_b = next(m for m in models if m['model'] == 'model_b')
        self.assertEqual(model_b['average_rank'], 2.0)
        self.assertEqual(model_b['win_rate'], 0.0)

    def test_logger(self):
        # Just verify logger exists and has handlers
        self.assertTrue(len(logger.handlers) > 0)
        logger.info("Test log message")

if __name__ == '__main__':
    unittest.main()
