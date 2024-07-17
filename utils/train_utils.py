from typing import Iterator, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def save_training_state(model: nn.Module, optimizer: Optimizer, iter_num: int, training_state_path: str) -> None:
    """Saves model state, optimizer state, and current iteration number.

    Args:
        model: model to be saved,
        optimizer: optimizer to be saved,
        iter_num: current iteration,
        training_state_path: path to save the current training state.
    """
    # TODO: consider map_location
    device = next(model.parameters()).device
    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iter_num,
    }, training_state_path)

    model.to(device)


def get_predictions(prediction_probabilities: np.ndarray) -> np.ndarray:
    """Collects prediction classes for given probabilities.

    Args:
        prediction_probabilities: array of shape (n_samples, n_classes).

    Returns:
        Array of predicted classes."""
    return np.array([
        np.argmax(probability)
        if torch.is_tensor(probability) else np.argmax(probability)
        for probability in prediction_probabilities
    ])


def get_next_batch(
    curr_iter: Iterator[Tuple[Tensor, Tensor]],
    data_loader: DataLoader = None,
) -> Tuple[Tensor, Tensor]:
    """Returns next batch of data in the iterator. If iterator ends, creates a new one.    

    Args:
        curr_iter: current iterator,
        data_loader: data loader used to create iterator.

    Returns:
        Tuple[Tensor, Tensor]: batch of the data: input and output data."""
    try:
        return next(curr_iter), curr_iter
    except StopIteration:
        curr_iter = iter(data_loader)
        return next(curr_iter), curr_iter
