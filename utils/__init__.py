import random
import numpy as np
import torch
import logging

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    return


def get_norm(x):
    """Compute the average Frobenius norm over a batch of tensors."""
    return torch.norm(x.view(x.shape[0], -1), p='fro', dim=-1).mean().item()


@torch.no_grad()
def batch_forward(model, x_mod, level, batch_size):
    """Forward pass in batches"""
    # x_mod: (n_samples, n_channels, image_size, image_size)
    grad=[]
    num_batches = (len(x_mod) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        if end_idx > len(x_mod):
            end_idx = len(x_mod)
        x = x_mod[start_idx:end_idx]
        labels = torch.ones(x.shape[0], device=x_mod.device) * level
        labels = labels.long()
        grad.append(model(x, labels))
    grad = torch.cat(grad, dim=0) # Shape: (n_samples, n_channels, image_size, image_size)
    return grad
