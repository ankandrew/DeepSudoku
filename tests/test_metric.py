import unittest
import torch
from deep_sudoku.metric import grid_accuracy, accuracy


class TestMetric(unittest.TestCase):
    def test_grid_accuracy(self):
        y_hat = torch.arange(4*9*9).reshape(4, 9, 9)
        y = y_hat.clone()
        # Without modifying y
        self.assertEqual(grid_accuracy(y_hat, y), 4)
        # Modifying one element
        y[1, 0, 2] = 1000
        self.assertEqual(grid_accuracy(y_hat, y), 3)

    def test_accuracy(self):
        y_hat = torch.ones(4, 9, 9)
        y = y_hat.clone()
        # Without modifying
        self.assertEqual(accuracy(y_hat, y), 4*9*9)
        # Modifying 2 random elements
        y[0, 0, 0] = -1
        y[2, 1, 4] = -1
        self.assertEqual(accuracy(y_hat, y), (4*9*9)-2)


if __name__ == '__main__':
    unittest.main()
