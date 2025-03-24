from datetime import datetime
from pathlib import Path
import inspect

import torch
from PIL import Image
import torchvision.transforms as T
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from data import ImageDataset
from model import EfficientNetB1
import model
from metrics import log_confusion, log_class_stats

torch.set_float32_matmul_precision('high')  # For my GPU


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
    class Trainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.train_outputs = []
            self.val_outputs = []

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, additional_losses = self.model(x)
            loss = self.loss_fn(y_hat, y)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            log_confusion(self, y_hat, y, "train")
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
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = ImageDataset("train", False, transform=get_transform(do_transform, res, crop, transform))
            self.label_names = dataset.label_names
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)

        def val_dataloader(self):
            dataset = ImageDataset("test", False, transform=get_transform(do_transform, res, crop, transform))
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
