import os
import builtins
import fcntl
import torch
import numpy as np
import random
import torch.distributed as dist
import functools
import socket
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from parallel_context import ParallelContext

def print(*args, **kwargs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def split_weight(data: torch.Tensor, dim: int) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = torch.chunk(data, world_size, dim=dim)
    return chunks[rank].contiguous()


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def global_wrapper(rank, func, tp, pp, dp, port, kwargs):
    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        # NOTE: since we do unit tests in a
        # single node => this is fine!
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp * pp * dp
    setup_dist_env(rank, world_size, port)
    parallel_context = ParallelContext(data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp)
    func(parallel_context, **kwargs)

def init_distributed(tp: int, dp: int, pp: int):
    def _init_distributed(func):
        def wrapper(**kwargs):

            world_size = tp * pp * dp
            port = find_free_port()

            # Note that kwargs needs to be passed as part of args in a way that can be unpacked
            args = (func, tp, pp, dp, port, kwargs)
            mp.spawn(global_wrapper, args=args, nprocs=world_size)

        return wrapper

    return _init_distributed

def assert_cuda_max_connections_set_to_1(func):
    flag_is_set_to_1 = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal flag_is_set_to_1
        if flag_is_set_to_1 is None:
            assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1"
            flag_is_set_to_1 = True
        return func(*args, **kwargs)

    return wrapper

class MNISTloader:
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data/",
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        train_val_split: float = 0.1,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_split = train_val_split

        self.setup()

    def setup(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        val_split = int(len(self.train_dataset) * self.train_val_split)
        train_split = len(self.train_dataset) - val_split

        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_split, val_split]
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

        print_rank_0(
            "Image Shape:    {}".format(self.train_dataset[0][0].numpy().shape),
            end="\n\n",
        )
        print_rank_0("Training Set:   {} samples".format(len(self.train_dataset)))
        print_rank_0("Validation Set: {} samples".format(len(self.val_dataset)))
        print_rank_0("Test Set:       {} samples".format(len(self.test_dataset)))

    def load(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        return train_loader, val_loader, test_loader
