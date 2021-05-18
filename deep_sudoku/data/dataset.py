import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

from .downloader import download_url
from .generator import Generator


class SudokuDataset(Dataset):
    def __init__(self, n=500, transform=None, a: float = 0.6, b: float = 0.8, from_generator: bool = True):
        """
        :param n: number of samples to generate
        :param transform: Optional transform to apply to features
        :param a: Lower bound probability for keeping values. Must be >= 0
        :param b: Upper bound probability for keeping values. Must be <= 1
        :param from_generator: Wether to generate Sudoku games from Generator

        Example: if a and b are set to 0.6 and 0.8 respectively, the probability
        of blanking (filling w/ zeros) cells at random will be 1 - 0.6 and 1 - 0.8.
        This means that in the worst case there will be 40% of missing cells.
        Note that this is sampled from an uniform distribution.
        """
        super(SudokuDataset, self).__init__()
        self.n = n
        if from_generator:
            self.x, self.y = Generator().generate_dataset(n, a, b)
        else:
            filename = '1m_sudoku_dataset_parsed.csv'
            ds_path = Path(__file__).parent / filename
            # Download if missing
            download_url(f'https://github.com/ankandrew/DeepSudoku/releases/download/v0.1-alpha/{filename}', ds_path)
            ds = pd.read_csv(ds_path, header=None, dtype='uint8').sample(frac=1).to_numpy()
            self.x, self.y = ds[:, :81].reshape(-1, 9, 9), ds[:, 81:].reshape(-1, 9, 9)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y
