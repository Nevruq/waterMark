from src.watermark import is_green
import unittest

class TestWatermarkLogitsManu(unittest.TestCase):

    def test_isgreen(self):
        # test 1:
        key = "ttlab123"
        vocab_size = 50
        context_ids = [1, 5, 8, 18, 10, 2, 7, 22, 2, 13]
        green_frac_1 = 0.3
        run_1 = is_green(vocab_size, context_ids, key, green_frac_1)
        run_2 = is_green(vocab_size, context_ids, key, green_frac_1)
        self.assertEqual(run_1, run_2)


if __name__ == "__main__":
    unittest.main()