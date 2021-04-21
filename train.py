import argparse
import copy
from timeit import default_timer as timer

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from deep_sudoku.data.dataset import SudokuDataset
from deep_sudoku.model import SudokuMLP, MultiBranchSudoku
from deep_sudoku.transform import ToTensor
from evaluate import eval_model
from utils import seed_all


def train(args, model, optimizer, loss_fn, train_loader, test_loader):
    """
    Train and validates a `model` with `optimizer` based on `train_loader` and
    `test_loader` respectively. Training and testing settings is based on `args`.

    :param args: Namespace containing config for epochs and device
    :param model: model to train
    :param optimizer: optimizer used to update parameters
    :param loss_fn: loss fn to be used
    :param train_loader: dataloader responsible for training
    :param test_loader: dataloader responsible for testing
    :return: Tuple of (model, optimizer) of the best test_grid_acc epoch
    """
    best_model, best_optimizer, best_accuracy = None, None, 0.0
    # Training loop
    # total_steps = len(train_loader)
    for epoch in range(args.epochs):
        # model.train()
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)
        start = timer()
        for i, (x, y) in enumerate(train_loader):
            # Move to corresponding device
            x = x.to(args.device)
            y = y.to(args.device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            y_hat = model(x)  # Out shape -> (batch, 729, 1, 1)
            # Reshape (batch, 729, 1, 1) -> (batch, 9, 9, 9)
            # Where dim=1 corresponds to the Class
            y_hat = y_hat.view(-1, 9, 9, 9)
            # Calculate loss
            loss = loss_fn(y_hat, y)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        # Validate accuracy
        model.eval()
        test_loss, test_acc, test_grid_acc = eval_model(args.device, model, test_loader, loss_fn)
        if test_grid_acc > best_accuracy:
            best_accuracy = test_grid_acc
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
        # Metrics
        train_loss, train_acc, train_grid_acc = eval_model(args.device, model, train_loader, loss_fn)
        print(f'loss {train_loss:.6f} accuracy {train_acc:.6f} grid_acc {train_grid_acc:.6f}\n'
              f'test_loss {test_loss:.6f} test_accuracy {test_acc:.10f} test_grid_acc {test_grid_acc:.6f}\n'
              f'Time: {(timer() - start):.3f} seconds')
    return best_model, best_optimizer


def main():
    # Make reproducible
    seed_all(1234)
    parser = argparse.ArgumentParser(description='Sudoku Model Training')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of epochs for model to train (default = 350)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate (default = 0.0001)')
    parser.add_argument('--batch', type=int, default=128, dest='batch_size',
                        help='Bactch size (default = 128)')
    parser.add_argument('--n-train', type=int, default=10_000,
                        help='Number of sudokus to generate for training (default = 10 000)')
    parser.add_argument('--n-test', type=int, default=2_500,
                        help='Number of sudokus to generate for test (default = 2 500)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to be used for training/testing (default = cpu)')
    args = parser.parse_args()

    # # Testing
    # args = parser.parse_args(
    #     [
    #         '--epochs', '25',
    #         '--lr', '1e-3',
    #         '--batch', '64',
    #         '--n-train', '500',
    #         '--n-test', '200',
    #         '--device', 'cpu'
    #     ]
    # )

    # Create datasets
    train_dataset = SudokuDataset(n=args.n_train, transform=ToTensor(one_hot=True), a=.55, b=.95)
    test_dataset = SudokuDataset(n=args.n_test, transform=ToTensor(one_hot=True), a=.55, b=.95)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    # model = SudokuMLP([10 * 9 * 9, 512, 9 * 9 * 9], batch_norm=True, dropout_rate=0.5).to(args.device)
    model = MultiBranchSudoku(input_channels=10).to(args.device)

    # Define Loss & Optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Pre-training
    print(f'{"#" * 40}\n--- Pre-Training ----\n{"#" * 40}')
    # Train loop
    start = timer()
    model2, optimizer2 = train(args, model, optimizer, loss_fn, train_loader, test_loader)
    print(f'Time taken {timer() - start} s')

    print(f'{"#" * 40}\n--- Training ----\n{"#" * 40}')
    train_dataset_2 = SudokuDataset(n=args.n_train, transform=ToTensor(one_hot=True), from_generator=False)
    test_dataset_2 = SudokuDataset(n=args.n_test, transform=ToTensor(one_hot=True), from_generator=False)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=args.batch_size, shuffle=False)
    start = timer()
    optimizer = optim.Adam(model2.parameters(), lr=1e-4)  # smaller lr
    optimizer.load_state_dict(optimizer2.state_dict())
    _, _ = train(args, model2, optimizer, loss_fn, train_loader_2, test_loader_2)
    print(f'Time taken {timer() - start} s')

    # Save best model
    torch.save(model2.state_dict(), './sudoku_model.pth')


if __name__ == '__main__':
    main()
