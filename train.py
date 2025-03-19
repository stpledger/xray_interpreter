from datetime import datetime
from pathlib import Path
from PIL import Image
import inspect
import torchvision.transforms as T
import model
import torch
torch.set_float32_matmul_precision('high') # For my GPU


models = {
    n: m for M in [model] for n, m in inspect.getmembers(M) if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}

def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def get_transform(do_transform: bool, resize: int = 256, crop: int = 240, norm: str = "efficientnet"):
    if do_transform:
        transform_list = [
            T.Resize(resize),
            T.CenterCrop(crop),
            T.Lambda(ensure_rgb),
            T.ToTensor()
        ]
        if norm == "efficientnet":
            transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        elif norm == "empirical":
            transform_list.append(T.Normalize(mean=[0.4985, 0.4985, 0.4985],
                                                std=[0.2493, 0.2493, 0.2493]))
        else:
            raise ValueError(f"Unknown normalization type: {norm}")
        return T.Compose(transform_list)
    else:
        return T.Compose([T.ToTensor()])

def train(model_name_or_path: str, epochs: int = 5, batch_size: int = 32, do_transform: bool = True, transform: str = "efficientnet", fresh: bool = False, res: int = 256, crop: int = 240):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    from data import ImageDataset
    from model import EfficientNetB1

    class Trainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.train_outputs = []
            self.val_outputs = []

        def log_confusion(self, y_hat, y, split: str):
            pred = (torch.sigmoid(y_hat) > 0.5).float()
            TP = ((pred == 1) & (y == 1)).sum()
            TN = ((pred == 0) & (y == 0)).sum()
            FP = ((pred == 1) & (y == 0)).sum()
            FN = ((pred == 0) & (y == 1)).sum()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            self.log(f'{split}/accuracy', accuracy)
            self.log(f'{split}/precision', precision)
            self.log(f'{split}/recall', recall)
            self.log(f'{split}/f1_score', f1_score)
            return
        
        def log_class_stats(self, y_hat, y, split: str):
            prob = torch.sigmoid(y_hat)
            pred = (prob > 0.5).float()
            for i, label in enumerate(self.label_names):
                TP = ((pred[:, i] == 1) & (y[:, i] == 1)).sum()
                TN = ((pred[:, i] == 0) & (y[:, i] == 0)).sum()
                FP = ((pred[:, i] == 1) & (y[:, i] == 0)).sum()
                FN = ((pred[:, i] == 0) & (y[:, i] == 1)).sum()
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                self.log(f'{label}/{split}/accuracy', accuracy)
                self.log(f'{label}/{split}/precision', precision)
                self.log(f'{label}/{split}/recall', recall)
                self.log(f'{label}/{split}/f1_score', f1_score)

                # Compute average probabilities for true and false labels
                true_mask = y[:, i] == 1
                false_mask = y[:, i] == 0
                if true_mask.sum() > 0:
                    avg_prob_true = prob[:, i][true_mask].mean()
                else:
                    avg_prob_true = torch.tensor(0.0, device=prob.device)
                if false_mask.sum() > 0:
                    avg_prob_false = prob[:, i][false_mask].mean()
                else:
                    avg_prob_false = torch.tensor(0.0, device=prob.device)
                self.log(f'{label}/{split}/avg_prob_true', avg_prob_true)
                self.log(f'{label}/{split}/avg_prob_false', avg_prob_false)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, additional_losses = self.model(x)
            loss = self.loss_fn(y_hat, y)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            self.log_confusion(y_hat, y, "train")
            self.train_outputs.append({"y_hat": y_hat, "y": y})
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            with torch.no_grad():
                y_hat, additional_losses = self.model(x)
                loss = self.loss_fn(y_hat, y)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            self.log_confusion(y_hat, y, "validation")
            self.val_outputs.append({"y_hat": y_hat, "y": y})
            return loss

        def on_train_epoch_end(self):
            aggregated_y_hat = torch.cat([out["y_hat"] for out in self.train_outputs], dim=0)
            aggregated_y = torch.cat([out["y"] for out in self.train_outputs], dim=0)
            self.log_class_stats(aggregated_y_hat, aggregated_y, "train")
            self.train_outputs.clear()

        def on_validation_epoch_end(self):
            aggregated_y_hat = torch.cat([out["y_hat"] for out in self.val_outputs], dim=0)
            aggregated_y = torch.cat([out["y"] for out in self.val_outputs], dim=0)
            self.log_class_stats(aggregated_y_hat, aggregated_y, "validation")
            self.val_outputs.clear()

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = ImageDataset("train", False, transform=get_transform(do_transform, res, crop, transform))
            self.label_names = dataset.label_names
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)

        def val_dataloader(self):
            dataset = ImageDataset("test", False, transform=get_transform(do_transform, res, crop, transform))
            self.label_names = dataset.label_names
            return torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)

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

    # Create the lightning model
    l_model = Trainer(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    trainer = L.Trainer(max_epochs=epochs, logger=logger, callbacks=[CheckPointer()])
    trainer.fit(
        model=l_model,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
