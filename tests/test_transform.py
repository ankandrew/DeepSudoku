import unittest
from deep_sudoku.transform import ToTensor
import numpy as np


class TestTransform(unittest.TestCase):
    def test_one_hot(self):
        simple_matrix = np.array([[2, 3, 0], [1, 0, 2]], dtype=np.int8)
        one_hot = ToTensor.one_hot_matrix(simple_matrix, calc_max=True)
        # Sum of channel dim should be 1
        equals = np.all(one_hot.sum(axis=0) == 1)
        self.assertTrue(equals)
        # Check if one_hot shape is (max_class+1, row, col)
        rows, cols = simple_matrix.shape
        self.assertTrue(list(one_hot.shape) == [simple_matrix.max() + 1, rows, cols])

    def test_call(self):
        lower_bound = np.random.randint(-1, 1)
        upper_bound = np.random.randint(1, 2)
        to_tensor = ToTensor([lower_bound, upper_bound])
        rand_array = np.random.randint(0, 10, (9, 9))
        norm_rand_array = to_tensor.normalize(rand_array)
        self.assertTrue(norm_rand_array.min() == lower_bound and upper_bound == norm_rand_array.max())


if __name__ == '__main__':
    unittest.main()
