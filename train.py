import torch
from torch import optim, nn
from deep_sudoku.data.dataset import SudokuDataset
from torch.utils.data import DataLoader
from deep_sudoku.transform import ToTensor
from deep_sudoku.model import SudokuMLP, MultiBranchSudoku
from timeit import default_timer as timer
from evaluate import eval_model
from utils import seed_all

# TESTING
cfg = {
    'epochs': 300,
    'lr': 1e-4,
    'batch_size': 64,
    'n_train': 10_000,
    'n_test': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    # 'device': 'cpu'
}


def train():
    train_dataset = SudokuDataset(n=cfg['n_train'], transform=ToTensor(one_hot=True))
    test_dataset = SudokuDataset(n=cfg['n_test'], transform=ToTensor(one_hot=True))

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    # Initialize model
    # model = SudokuMLP([10 * 9 * 9, 120, 120, 9 * 9 * 9], batch_norm=False, dropout_rate=0.5).to(cfg['device'])
    model = MultiBranchSudoku(input_channels=10).to(cfg['device'])
    # Define Loss / Optimizer
    optimizer = optim.Adam(model.parameters(), cfg['lr'])
    loss = nn.CrossEntropyLoss()
    # Training loop
    # total_steps = len(train_loader)
    for epoch in range(cfg['epochs']):
        # model.train()
        print(f'Epoch {epoch}/{cfg["epochs"] - 1}')
        print('-' * 10)
        start = timer()
        for i, (x, y) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Move to corresponding device
            x = x.to(cfg['device'])
            y = y.to(cfg['device'])
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
        test_loss, test_acc, test_grid_acc = eval_model(cfg['device'], model, test_loader, loss)
        # Metrics
        train_loss, train_acc, train_grid_acc = eval_model(cfg['device'], model, train_loader, loss)
        print(f'loss {train_loss:.6f} accuracy {train_acc:.6f} grid_acc {train_grid_acc:.6f}\n'
              f'test_loss {test_loss:.6f} test_accuracy {test_acc:.6f} test_grid_acc {test_grid_acc:.6f}\n'
              f'Time: {(timer() - start):.3f} seconds')
    return model


if __name__ == '__main__':
    # Make reproducible
    seed_all(1234)

    start = timer()
    model = train()
    print(f'Time taken {timer() - start} s')
    # TODO: save / save best model
    # torch.save(model.state_dict(), './sudoku_model.pth')
