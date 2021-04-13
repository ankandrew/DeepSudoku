import argparse
import copy
from timeit import default_timer as timer

from torch import optim, nn
from torch.utils.data import DataLoader

from deep_sudoku.data.dataset import SudokuDataset
from deep_sudoku.model import SudokuMLP
from deep_sudoku.transform import ToTensor
from evaluate import eval_model
from utils import seed_all


def train(args):
    train_dataset = SudokuDataset(n=args.n_train, transform=ToTensor(one_hot=True))
    test_dataset = SudokuDataset(n=args.n_test, transform=ToTensor(one_hot=True))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model, best_accuracy = None, 0.0

    # Initialize model
    model = SudokuMLP([10 * 9 * 9, 512, 9 * 9 * 9], batch_norm=True, dropout_rate=0.5).to(args.device)
    # model = MultiBranchSudoku(input_channels=10).to(args.device)
    # Define Loss / Optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    loss = nn.CrossEntropyLoss()
    # Training loop
    # total_steps = len(train_loader)
    for epoch in range(args.epochs):
        # model.train()
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)
        start = timer()
        for i, (x, y) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Move to corresponding device
            x = x.to(args.device)
            y = y.to(args.device)
            # Forward pass
            y_hat = model(x)  # Out shape -> (batch, 729, 1, 1)
            # Reshape (batch, 729, 1, 1) -> (batch, 9, 9, 9)
            # Where dim=1 corresponds to the Class
            y_hat = y_hat.view(-1, 9, 9, 9)
            # Calculate loss
            output = loss(y_hat, y)
            # Backward pass
            output.backward()
            # Update weights
            optimizer.step()
        # Validate accuracy
        model.eval()
        test_loss, test_acc, test_grid_acc = eval_model(args.device, model, test_loader, loss)
        if test_grid_acc > best_accuracy:
            best_accuracy = test_grid_acc
            best_model = copy.deepcopy(model)
        # Metrics
        train_loss, train_acc, train_grid_acc = eval_model(args.device, model, train_loader, loss)
        print(f'loss {train_loss:.6f} accuracy {train_acc:.6f} grid_acc {train_grid_acc:.6f}\n'
              f'test_loss {test_loss:.6f} test_accuracy {test_acc:.6f} test_grid_acc {test_grid_acc:.6f}\n'
              f'Time: {(timer() - start):.3f} seconds')
    return best_model


if __name__ == '__main__':
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
    # # Testing
    # args = parser.parse_args(
    #     [
    #         '--epochs', '100',
    #         '--lr', '1e-3',
    #         '--batch', '128',
    #         '--n-train', '50_000',
    #         '--n-test', '2_000',
    #         '--device', 'cuda'
    #     ]
    # )
    # # args = parser.parse_args()
    start = timer()
    model = train(args)
    print(f'Time taken {timer() - start} s')
    # torch.save(model.state_dict(), './sudoku_model.pth')
