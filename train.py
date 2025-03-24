from datetime import datetime
from pathlib import Path
import inspect
import math
import pandas as pd

import torch
from PIL import Image
import torchvision.transforms as T
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics

from data import ImageDataset
import model
from transforms import get_transform
from metrics import log_confusion, log_class_stats

torch.set_float32_matmul_precision('high')  # For my GPU


models = {
    n: m for M in [model] for n, m in inspect.getmembers(M) if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}

def train(model_name_or_path: str, 
          epochs: int = 5, 
          batch_size: int = 32,
          lr: float = 1e-3, 
          warmup: float = 3.0,
          weighted: bool = True,
          fresh: bool = False, 
          do_transform: bool = True, 
          transform: str = "efficientnet", 
          res: int = None):
    
    class Trainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

            if weighted:
                df = pd.read_csv('positive_weights.csv')
                pos_weights = torch.tensor(df["Positive Weights"].values, dtype=torch.float)
                self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            
            self.train_outputs = []
            self.val_outputs = []
            # Initialize aggregated metrics for training.
            self.train_precision = torchmetrics.Precision(task="multilabel", num_labels=14, threshold=0.5)
            self.train_recall = torchmetrics.Recall(task="multilabel", num_labels=14, threshold=0.5)
            self.train_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=14, threshold=0.5)
            self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=14, average="macro", threshold=0.5)
            # Initialize loss accumulator
            self.train_loss_sum = 0.0
            self.train_loss_count = 0
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, additional_losses = self.model(x)
            loss = self.loss_fn(y_hat, y)
            
            # Update loss accumulator.
            self.train_loss_sum += loss.item()
            self.train_loss_count += 1
            
            # Log per-step losses from additional_losses if desired.
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            
            # log_confusion(self, y_hat, y, "train")
            self.train_outputs.append({"y_hat": y_hat, "y": y})
            
            # Compute predictions with a threshold of 0.5.
            preds = (torch.sigmoid(y_hat) > 0.5).int()
            targets = y.int()
            # Update each metric.
            self.train_precision.update(preds, targets)
            self.train_recall.update(preds, targets)
            self.train_accuracy.update(preds, targets)
            self.train_f1.update(preds, targets)
    
            # Every 50 steps, log aggregated loss and metrics, then reset accumulators.
            if (batch_idx + 1) % 50 == 0:
                avg_loss = self.train_loss_sum / self.train_loss_count
                self.log("train/loss", avg_loss, on_step=True, prog_bar=True)
                self.log("train/precision", self.train_precision.compute(), on_step=True)
                self.log("train/recall", self.train_recall.compute(), on_step=True)
                self.log("train/accuracy", self.train_accuracy.compute(), on_step=True)
                self.log("train/f1", self.train_f1.compute(), on_step=True)
                # Reset all metrics and loss accumulation
                self.train_precision.reset()
                self.train_recall.reset()
                self.train_accuracy.reset()
                self.train_f1.reset()
                self.train_loss_sum = 0.0
                self.train_loss_count = 0
    
            return loss
    
        def validation_step(self, batch, batch_idx):
            x, y = batch
            with torch.no_grad():
                y_hat, additional_losses = self.model(x)
                loss = self.loss_fn(y_hat, y)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            log_confusion(self, y_hat, y, "validation")
            self.val_outputs.append({"y_hat": y_hat, "y": y})
            return loss
    
        def on_train_epoch_end(self):
            aggregated_y_hat = torch.cat([out["y_hat"] for out in self.train_outputs], dim=0)
            aggregated_y = torch.cat([out["y"] for out in self.train_outputs], dim=0)
            log_class_stats(self, aggregated_y_hat, aggregated_y, self.label_names, "train")
            self.train_outputs.clear()
    
        def on_validation_epoch_end(self):
            aggregated_y_hat = torch.cat([out["y_hat"] for out in self.val_outputs], dim=0)
            aggregated_y = torch.cat([out["y"] for out in self.val_outputs], dim=0)
            log_class_stats(self, aggregated_y_hat, aggregated_y, self.label_names, "validation")
            self.val_outputs.clear()
    
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
            total_steps = self.trainer.estimated_stepping_batches
            # Define warm-up to last for 3 epochs:
            warmup_steps = int(total_steps * (warmup / epochs))
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1. + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
        def train_dataloader(self):
            dataset = ImageDataset("train", False, transform=get_transform(do_transform, model_name, res, transform))
            self.label_names = dataset.label_names
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)
    
        def val_dataloader(self):
            dataset = ImageDataset("test", False, transform=get_transform(do_transform, model_name, res, transform))
            self.label_names = dataset.label_names
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size*4, num_workers=4, shuffle=False)
    
    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            fn = Path(f"checkpoints/{timestamp}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)
            torch.save(model, Path(__file__).parent / f"{model_name}.pth")
    
    # Load or create the model
    if Path(model_name_or_path).exists():
        model = torch.load(model_name_or_path, weights_only=False)
        model_name = model.__class__.__name__
    else:
        model_name = model_name_or_path
        if model_name in models:
            model = models[model_name](num_classes=14, fresh=fresh)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    l_model = Trainer(model)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    trainer = L.Trainer(max_epochs=epochs, logger=logger, callbacks=[CheckPointer()])
    trainer.fit(model=l_model)
    
    
if __name__ == "__main__":
    from fire import Fire
    Fire(train)
