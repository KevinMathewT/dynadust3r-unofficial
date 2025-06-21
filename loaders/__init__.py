import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from loaders.pointodyssey import PointOdyssey
from loaders.stereo4d import Stereo4D
from loaders.stereo4dv2 import Stereo4Dv2
from loaders.stereo4dv3 import Stereo4Dv3
from loaders.stereo4dv4 import Stereo4Dv4
from loaders.stereo4dv5 import Stereo4Dv5

LOADERS = {
    "pointodyssey": PointOdyssey,
    # "stereo4d": Stereo4D,
    # "stereo4d": Stereo4Dv2,
    # "stereo4d": Stereo4Dv3,
    # "stereo4d": Stereo4Dv4,
    "stereo4d": Stereo4Dv5,
}

def add_batch_size_wrapper(batch):
    batch = default_collate(batch)
    batch['batch_size'] = len(batch['left_pm'])  # or use any other tensor in the batch
    
    return batch

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
        collate_fn=add_batch_size_wrapper,
    )

    valid_dataset = LOADERS[config.data.loader](config, valid=True)
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.batch_size // world_size,
        shuffle=False,  # validation data should not be shuffled
        num_workers=config.data.num_workers,
        sampler=valid_sampler,
        collate_fn=add_batch_size_wrapper,
    )

    # print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Train dataloader size: {len(train_loader)}")
    print(f"Validation dataloader size: {len(valid_loader)}")

    # Debug print after loaders are created
    print(f"Global batch size: {config.data.batch_size}")
    if train_sampler:
        print(f"Using DistributedSampler for train_loader")
    else:
        print(f"Using SequentialSampler for train_loader")

    for batch in train_loader:
        batch_keys = list(batch.keys())  # Dynamically check available keys
        print(f"Local train batch size per GPU: {batch[batch_keys[0]].size(0)}")
        break

    for batch in valid_loader:
        batch_keys = list(batch.keys())  # Dynamically check available keys
        print(f"Local valid batch size per GPU: {batch[batch_keys[0]].size(0)}")
        break
    
    return train_loader, valid_loader