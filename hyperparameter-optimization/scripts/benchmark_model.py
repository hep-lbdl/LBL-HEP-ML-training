import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import energyflow

from torch.utils.data import TensorDataset, random_split, DataLoader
from utils import make_mlp
from sklearn.metrics import roc_auc_score


class BenchmarkClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)

        self.layers = make_mlp(
            input_size = 45,
            sizes = [self.hparams["n_channels"]] * self.hparams["n_layers"] + [1],
            hidden_activation = self.hparams["hidden_activation"],
            output_activation = None,
            layer_norm = self.hparams["layer_norm"],
            batch_norm = self.hparams["batch_norm"]                                    
        )
        
    def setup(self, stage):
        
        all_jets = energyflow.qg_nsubs.load(num_data=-1, cache_dir=self.hparams["input_directory"])
        dataset = TensorDataset(*[torch.from_numpy(tensor).float() for tensor in all_jets])
        self.trainset, self.valset, self.testset = random_split(dataset, self.hparams["train_split"])
        
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch_size"])
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1000)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1000)
        else:
            return None
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                        self.parameters(),
                        lr=(self.hparams["lr"]),
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        amsgrad=True,
                    )
        
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=self.hparams["patience"],
                            gamma=self.hparams["factor"],
                        ),
                        "interval": "epoch",
                        "frequency": 1,
                    }
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        
        input_data, y = batch
        output = self(input_data).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(output, y, pos_weight = torch.tensor(self.hparams["weight"]))
        
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True)
        
        return loss
        
    def get_metrics(self, batch, truth, output):
        
        predictions = torch.sigmoid(output) > self.hparams["edge_cut"]
        
        edge_positive = predictions.sum().float()
        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & predictions).sum().float()        

        eff = edge_true_positive / edge_true
        pur = edge_true_positive / edge_positive

        auc = roc_auc_score(truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach())
        
        return predictions, eff, pur, auc
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        
        input_data, y = batch
        output = self(input_data).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(output, y)
        
        predictions, eff, pur, auc = self.get_metrics(batch, y, output)
        
        metrics = {"val_loss": loss, "eff": eff, "pur": pur, "auc": auc}
        self.log_dict(metrics)
        
        return metrics
    
    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["val_loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs
    
    def forward(self, x):
        
        x = self.layers(x)
        
        return x