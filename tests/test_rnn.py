import unittest

import torch

from deep_sudoku.model.model import RNN


class TestRNN(unittest.TestCase):
    def setUp(self):
        self.hidden, self.out_classes = 64, 9
        self.batch, self.seq_len, self.input_size = 16, 81, 10
        self.random_input = torch.randn((self.batch, self.seq_len, self.input_size), dtype=torch.float32)

    def test_unidirectional(self):
        rnn = RNN(self.input_size, self.hidden, 1, 'lstm', [self.hidden, self.out_classes],
                  fc_bn=True, fc_dropout=0.25, bidirectional=False)
        out = rnn(self.random_input)
        self.assertEqual(out.shape, (self.batch, self.seq_len, self.out_classes))

    def test_bidirectional(self):
        rnn = RNN(self.input_size, self.hidden, 1, 'lstm', [self.hidden * 2, self.out_classes],
                  fc_bn=True, fc_dropout=0.25, bidirectional=True)
        out = rnn(self.random_input)
        self.assertEqual(out.shape, (self.batch, self.seq_len, self.out_classes))

    def test_no_fully_connected(self):
        for bidirec, rnn_type in ((True, 'lstm'), (False, 'gru'), (True, 'rnn')):
            rnn = RNN(self.input_size, self.hidden, 1, rnn_type, fc_layers=None, bidirectional=bidirec)
            out = rnn(self.random_input)
            self.assertEqual(out.shape, (self.batch, self.seq_len, self.hidden * 2 if bidirec else self.hidden))


if __name__ == '__main__':
    unittest.main()
