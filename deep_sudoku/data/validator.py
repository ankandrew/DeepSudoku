import numpy as np


class Validator:
    def __init__(self):
        self.to_compare = np.arange(1, 10)

    def __call__(self, grid):
        return self.validate(grid)

    def validate(self, grid: np.ndarray) -> bool:
        valid = True
        if not self.val_sub_grids(grid):
            valid = False
        # Validate columns
        if not self.validate_axis(grid, axis=0):
            valid = False
        # Validate rows
        if not self.validate_axis(grid, axis=1):
            valid = False
        return valid

    def val_sub_grids(self, grid: np.ndarray) -> bool:
        """
        Validate each 3x3 subgrid of the 9x9 grid

        :param grid: grid of type ndarray filled with all the values
        :return: True if all subgrids contains the number [0, 9] w/o dups
        """
        # Change so that each row contains 3x3 grid
        grid = self.blockshaped(grid, 3, 3)
        # Sort
        grid = np.sort(grid, axis=1)
        # Compare if values are the same as [1, 2, ..., 9]
        mask = grid == self.to_compare
        # All must be true
        return np.all(mask)

    def validate_axis(self, grid: np.ndarray, axis: int) -> bool:
        """
        Validate that all rows(axis=1) or columns(axis=0)
        have values of [0, 9] with out duplicates

        :param grid: Array containing the grid fulled with values
        :param axis: Number 0 or 1, wether to check column/row - wise
        :return: True if all columns/rows contains numbers from [0, 9] w/o dups
        """
        if axis == 1:
            grid = np.sort(grid, axis=1)
            return np.all(grid == self.to_compare)
        elif axis == 0:
            grid = np.sort(grid, axis=0)
            return np.all(grid == np.expand_dims(self.to_compare, axis=0).T)
        else:
            raise ValueError('Axis must be equal to 0 or 1')

    @staticmethod
    def blockshaped(arr: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
        """
        Modified from https://stackoverflow.com/a/16858283

        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows * ncols))
