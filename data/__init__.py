
#
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
#
# from utils.utils import is_distributed
# from .mot_dataset import build as build_mot_dataset
# from .utils import collate_fn


from torch.utils.data import DataLoader

from .joint_dataset import JointDataset
from .transforms import build_transforms
from .util import collate_fn


def build_dataset(config: dict):
    return JointDataset(
        data_root=config["DATA_ROOT"],
        datasets=config["DATASETS"],
        splits=config["DATASET_SPLITS"],
        transforms=build_transforms(config),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

# # For MOT17
# def build_dataset(config: dict):
#     return build_mot_dataset(config=config)
#
#
# def build_sampler(dataset, shuffle: bool):
#     if is_distributed():
#         sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
#     else:
#         sampler = RandomSampler(dataset) if shuffle is True else SequentialSampler(dataset)
#     return sampler
#
#
# def build_dataloader(dataset, sampler, batch_size: int, num_workers: int):
#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True
#     )