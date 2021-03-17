import torch


class ToTensor:
    """
    Convert to tensorts X / y and rescales X values
    """
    def __init__(self, scale: list):
        if len(scale) != 2:
            raise ValueError('Scale len must be 2')
        self.scale = scale

    def __call__(self, x, y):
        # Convert to Tensors
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        # Re-scale x
        x = self.__normalize(x)
        # Add channel dimension
        x = torch.unsqueeze(x, dim=0)
        # Make y class index of range [0, C−1]
        y -= 1
        return x, y

    def __normalize(self, x):
        """
        Normalize values between [a, b] / [self.scale[0], [self.scale[1]]

        :param x: Tensor to normalize
        :return: Normalized tensor between [self.scale[0], [self.scale[1]]
        """
        # min_num, max_num = x.max().item(), x.min().item()
        # prob = (x - min_num) / (max_num - min_num)
        return (self.scale[1] - self.scale[0]) * (x / 9) + self.scale[0]


# Debug

# trans = ToTensor([-1, 1])
# x = torch.tensor([
#     [0, 0, 4, 2, 3, 0, 0, 0, 7],
#     [0, 0, 0, 1, 8, 0, 0, 5, 0],
#     [1, 0, 0, 0, 0, 0, 2, 0, 0],
#     [8, 7, 0, 0, 4, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 7, 6],
#     [0, 0, 0, 0, 7, 0, 0, 0, 2],
#     [0, 0, 0, 0, 0, 0, 0, 0, 3],
#     [0, 0, 0, 0, 0, 0, 0, 0, 8],
#     [0, 2, 0, 9, 1, 8, 0, 6, 5]], dtype=torch.float32
# )
# trans.normalize(x)
