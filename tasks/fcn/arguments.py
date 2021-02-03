from .dataset import add_dataset_args
from .model import add_model_args


def add_task_args(parser):
    add_dataset_args(parser)
    add_model_args(parser)
