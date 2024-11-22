#!/usr/bin/env python
import os
import torch

from lightning.pytorch.cli import LightningCLI

from models.task import DETRDetection
from datasets.coco_data import COCO

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_config_file = os.path.join(current_dir, "fit.yaml")
    cli = LightningCLI(
        DETRDetection,
        COCO,
        parser_kwargs={
            "fit": {"default_config_files": [base_config_file]},
            "validate": {"default_config_files": [base_config_file]},
        },
    )
