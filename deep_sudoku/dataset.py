from torch.utils.data import Dataset
from .generator import Generator


class SudokuDataset(Dataset):
    def __init__(self, n=500, transform=None):
        """
        :param n: number of samples to generate
        :param transform: Optional transform to apply to features
        """
        super(SudokuDataset, self).__init__()
        self.n = n
        self.x, self.y = Generator().generate_dataset(n)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y
