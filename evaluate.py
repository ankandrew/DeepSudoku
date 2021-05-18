from deep_sudoku.metric import grid_accuracy, accuracy
from typing import Tuple
import torch


def eval_model(device, model, dataloader, loss_fn) -> Tuple[float, float, float]:
    loss, acc, grid_acc = 0.0, 0.0, 0.0
    # total = cfg['n_test']
    total = len(dataloader.dataset)
    with torch.no_grad():
        for x, y in dataloader:
            # Move to corresponding device
            x = x.to(device)
            x = x.reshape((x.size(0), 81, 10))  # For RNN
            y = y.to(device)
            # Forward
            y_hat_test = model(x)  # Out shape -> (batch, 729, 1, 1)
            y_hat_test = y_hat_test.view(-1, 9, 9, 9)
            # Calculate Loss
            output_loss = loss_fn(y_hat_test, y)
            loss += output_loss.item() * x.size(0)
            # Find the corresponding class
            _, y_hat_test = y_hat_test.max(dim=1)  # Out shape -> (batch, 9, 9)
            # Check accuracy
            grid_acc += grid_accuracy(y_hat_test, y, valid=False)  # Speed-up training valid=False
            acc += accuracy(y_hat_test, y)
        loss /= total
        grid_acc /= total
        # For each slot in all training grids
        acc /= (total * 9 * 9)
    return loss, acc, grid_acc
