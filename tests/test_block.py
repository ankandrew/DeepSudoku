import unittest
from deep_sudoku.model.block import FullyConnected
import torch


class TestModel(unittest.TestCase):
    def test_fc(self):
        n_logits, n_input = 6, 10
        fc = FullyConnected([n_input, 4, 4, n_logits])
        random_input = torch.rand(32, n_input)
        self.assertEqual(fc(random_input).size(-1), n_logits)


if __name__ == '__main__':
    unittest.main()
