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

transform_b1 = T.Compose([
        T.Resize(256),
        T.CenterCrop(240),
        T.Lambda(ensure_rgb),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # EfficientNet's expectations
    ])

def train(model_name_or_path: str, epochs: int = 5, batch_size: int = 64, transform=transform_b1):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    from data import ImageDataset
    from model import EfficientNetB1

    class Trainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, additional_losses = self.model(x)
            loss = self.loss_fn(y_hat, y)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            with torch.no_grad():
                y_hat, additional_losses = self.model(x)
                loss = self.loss_fn(y_hat, y)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            # if batch_idx == 0:
            #     self.logger.experiment.add_images(
            #         "input", (x[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
            #     )
            #     self.logger.experiment.add_images(
            #         "prediction", (x_hat[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
            #     )
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = ImageDataset("train", False, transform=transform)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)

        def val_dataloader(self):
            dataset = ImageDataset("test", False, transform=transform)
            return torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=4, shuffle=False)

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
            model = models[model_name](num_classes=14)
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
