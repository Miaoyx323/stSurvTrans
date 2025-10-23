from .dataset import vaeDataset
import torch.utils.data as Data

def csDataLoader(
    dataset: vaeDataset, 
    batch_size: int = 128, 
    sampler = None, 
    shuffle: bool = True
):
    return Data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler
    )