import os
import torch
from tap import Tap
from pytorch_lightning import Trainer, seed_everything

from model import *


class ArgParser(Tap):
    lr: float = 2e-5  # Starting Learning Rate
    epochs: int = 20  # Max Epochs
    batch_size: int = 32  # Train/Eval Batch Size
    random_seed: int = 42  # Random Seed
    train_data_path: str = None  # Train Dataset file path. csv, tsv, xlsx
    val_data_path: str = None  # Validation Dataset file path. csv, tsv, xlsx
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    fp16: bool = False  # Enable train on FP16

    input_type: str = 'FCN'  # Model name
    nlayers: int = 3  # Number of layers
    nhids: int = 256  # Number of hidden cells
    input_length: int = 100  # Input dimension


if __name__ == "__main__":
    print("Using PyTorch Ver", torch.__version__)

    args = ArgParser().parse_args()
    # args.save("args.json")

    print("Fix Seed: ", args.random_seed)
    seed_everything(args.random_seed)

    model = eval(f"{args.input_type}Model")(args)

    print(":: Start Training ::")
    trainer = Trainer(
        deterministic=True,
        max_epochs=args.epochs,
        gpus=torch.cuda.device_count(),
        num_sanity_val_steps=0,
        precision=16 if args.fp16 else 32,
        distributed_backend='dp'
    )

    trainer.fit(model)
