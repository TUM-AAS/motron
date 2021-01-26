from typing import Callable
import numpy as np
import torch
from torch.distributions import Distribution


def mutual_inf_px(p_yz: Distribution):
    dist = p_yz.__class__
    H_y = dist(probs=p_yz.probs.mean(dim=0)).entropy()
    return (H_y - p_yz.entropy().mean(dim=0)).mean()


def get_activation_function(activation_function) -> Callable:
    if activation_function is None or activation_function == "":
        return lambda x: x
    elif type(activation_function) == str:
        try:
            return getattr(torch, activation_function)
        except AttributeError:
            return getattr(torch.nn, activation_function)()
    elif callable(activation_function):
        return activation_function


def to_dtype(model, dtype: str):
    return getattr(model, dtype)()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    np.random.seed(seed)