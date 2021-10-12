import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

# ----------------
# Main train function
# ----------------
from argparse import ArgumentParser

def main():
    
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--conda_env", type=str, default="some_name")
    parser.add_argument("--notification_email", type=str, default="will@email.com")

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

if __init__ = "__main__":
    main()