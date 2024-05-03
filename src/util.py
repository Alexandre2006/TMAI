import torch
import numpy as np
from typing import Sequence, Mapping

# Written by GitHub Copilot
def collate(batch, device):
    first_elem = batch[0]

    if isinstance(first_elem, torch.Tensor):
        return torch.stack([b.contiguous().to(device) for b in batch], 0)

    elif isinstance(first_elem, np.ndarray):
        return collate(tuple(torch.from_numpy(b) for b in batch), device)

    elif hasattr(first_elem, '__torch_tensor__'):
        return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
    
    elif isinstance(first_elem, Sequence):
        transposed_batch = zip(*batch)
        return type(first_elem)(collate(samples, device) for samples in transposed_batch)

    elif isinstance(first_elem, Mapping):
        return type(first_elem)((key, collate(tuple(d[key] for d in batch), device)) for key in first_elem)
    
    else:
        return torch.from_numpy(np.array(batch)).to(device)