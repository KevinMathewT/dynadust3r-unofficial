import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from loaders.pointodyssey import PointOdyssey

LOADERS = {
    "pointodyssey": PointOdyssey,
}

def get_loaders(config):
    train_dataset = LOADERS[config.data.loader](config, valid=False)

    is_distributed = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_distributed else 1

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size // world_size,  # Scale batch size for distributed
        shuffle=(train_sampler is None),  # Shuffle only if not using sampler
        num_workers=config.data.num_workers,
        sampler=train_sampler,
    )

    valid_dataset = PointOdyssey(config, valid=True)
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.batch_size // world_size,
        shuffle=False,  # validation data should not be shuffled
        num_workers=config.data.num_workers,
        sampler=valid_sampler,
    )

    # Debug print after loaders are created
    print(f"Global batch size: {config.data.batch_size}")
    if train_sampler:
        print(f"Using DistributedSampler for train_loader")
    else:
        print(f"Using SequentialSampler for train_loader")

    for batch in train_loader:
        batch_keys = list(batch.keys())  # Dynamically check available keys
        print(f"Local batch size per GPU: {batch['image'][batch_keys[0]].size(0)}")
        break

    return train_loader, valid_loader