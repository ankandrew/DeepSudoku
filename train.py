import torch
from torch import optim, nn
from deep_sudoku.dataset import SudokuDataset
from torch.utils.data import DataLoader
from deep_sudoku.transform import ToTensor
from deep_sudoku.model import SudokuModel
from timeit import default_timer as timer
from deep_sudoku.metric import grid_accuracy, accuracy

# from tqdm import tqdm

cfg = {
    'epochs': 5_00,
    'lr': 1e-3,
    'batch_size': 64,
    'n_train': 20_000,
    'n_test': 2_000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def test_score(model, dataloader, loss_fn) -> tuple[float, float, float]:
    loss, acc, grid_acc = 0.0, 0.0, 0.0
    # total = cfg['n_test']
    total = len(dataloader.dataset)
    with torch.no_grad():
        for x, y in dataloader:
            # Move to corresponding device
            x = x.to(cfg['device'])
            y = y.to(cfg['device'])
            # Forward
            y_hat_test = model(x)  # Out shape -> (batch, 729, 1, 1)
            y_hat_test = y_hat_test.view(-1, 9, 9, 9)
            # Calculate Loss
            output_loss = loss_fn(y_hat_test, y).item()
            loss += output_loss
            # Find the corresponding class
            _, y_hat_test = y_hat_test.max(dim=1)  # Out shape -> (batch, 9, 9)
            # Check accuracy
            grid_acc += grid_accuracy(y_hat_test, y)
            acc += accuracy(y_hat_test, y)
    loss /= total
    grid_acc /= total
    # For each slot in all training grids
    # TODO: Make generic for n x n grid
    acc /= (total * 9 * 9)
    return loss, acc, grid_acc


def main() -> None:
    train_dataset = SudokuDataset(n=cfg['n_train'], transform=ToTensor([0, 1]))
    test_dataset = SudokuDataset(n=cfg['n_test'], transform=ToTensor([0, 1]))

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    # print(f'ds[0]: {train_dataset[0]}')
    # print(f'ds[400]: {train_dataset[40]}')

    # Initialize model
    model = SudokuModel().to(cfg['device'])
    # Define Loss / Optimizer
    optimizer = optim.Adam(model.parameters(), cfg['lr'])
    loss = nn.CrossEntropyLoss()
    # Training loop
    total_steps = len(train_loader)
    train_acc, train_grid_acc = 0, 0
    for epoch in range(cfg['epochs']):
        # time_per_epoch = 0
        start = timer()
        for i, (x, y) in enumerate(train_loader):
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
            optimizer.zero_grad()
            output.backward()
            # Update weights
            optimizer.step()
            # Validate accuracy
            model.eval()
            test_loss, test_acc, test_grid_acc = test_score(model, test_loader, loss)
            model.train()
            # Metrics
            with torch.no_grad():
                _, y_hat = y_hat.max(dim=1)
                train_grid_acc += grid_accuracy(y_hat, y)
                train_acc += accuracy(y_hat, y)
            # Print stats
            # if (i + 1) % (total_steps // 2) == 0:
            #     print(f'Epoch {epoch}/{cfg["epochs"]}\t {(i + 1)}/{total_steps}\t loss {output:.6f}\t'
            #           f'val_loss {test_loss:.6f}\t test_accuracy {test_acc:.6f}\t test_grid_acc {test_grid_acc:.6f}')
        train_acc /= len(train_loader.dataset)
        train_grid_acc /= (len(train_loader.dataset) * 9 * 9)
        print(f'Epoch {epoch}/{cfg["epochs"]}\t loss {output:.6f}\t'
              f'train_accuracy {train_acc:.6f}\t train_grid_acc {train_grid_acc:.6f}\t'
              f'val_loss {test_loss:.6f}\t test_accuracy {test_acc:.6f}\t test_grid_acc {test_grid_acc:.6f}\t'
              f'Epoch {epoch} took {(timer() - start):.3f} seconds')


if __name__ == '__main__':
    main()
