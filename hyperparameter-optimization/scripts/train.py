import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from benchmark_model import BenchmarkClassifier

import wandb


def main():
    
    wandb.init()
    config = wandb.config
    
    model = BenchmarkClassifier(dict(config))
    logger = WandbLogger()
    trainer = Trainer(gpus=1, max_epochs=config["max_epochs"], logger=logger)
    trainer.fit(model)
    
    
if __name__ == "__main__":
    main()