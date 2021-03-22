import unittest
from deep_sudoku.data.dataset import SudokuDataset
from torch.utils.data import DataLoader
from deep_sudoku.transform import ToTensor
import torch


class DatasetTest(unittest.TestCase):
    def test_sudoku_ds(self):
        ds = SudokuDataset(n=1_000, transform=ToTensor(one_hot=True))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        epochs_test = 5
        for _ in range(epochs_test):
            for x, y in loader:
                self.assertTrue(torch.all(x.sum(dim=1).type(torch.bool)))
                # Y should range from (0, C-1) == (0, 8)
                self.assertEqual(y.max().item(), 8)
                self.assertEqual(y.min().item(), 0)


if __name__ == '__main__':
    unittest.main()
