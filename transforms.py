from PIL import Image
import torchvision.transforms as T

def ensure_rgb(img: Image.Image) -> Image.Image:
    """Converts an image to RGB if needed."""
    return img.convert("RGB") if img.mode != "RGB" else img

def get_transform(do_transform: bool,
                  model_name: str = None,
                  resize: int = None,
                  norm: str = "efficientnet"):
    """
    Returns a torchvision Compose transform based on parameters.

    Parameters:
        do_transform (bool): Whether to apply a full transform pipeline or just convert to tensor.
        model_name (str): Name of the model. If the model is an EfficientNet, a default resize is used if not provided.
        resize (int): The target size for resizing the image. Overrides the default EfficientNet value if provided.
        norm (str): Normalization type to use: "efficientnet" or "empirical".

    Returns:
        A torchvision.transforms.Compose object.
    """
    if do_transform:
        # If model_name indicates an EfficientNet, set default resize if not provided.
        if model_name is not None and "efficientnet" in model_name.lower():
            efficientnet_defaults = {
                "EfficientNetB0": 224,
                "EfficientNetB1": 240,
                "EfficientNetB2": 260,
                "EfficientNetB3": 300,
                "EfficientNetB4": 380,
                "EfficientNetB5": 456,
                "EfficientNetB6": 528,
                "EfficientNetB7": 600
            }
            key = model_name.lower()
            if key in efficientnet_defaults and resize is None:
                resize = efficientnet_defaults[key]
        # Set safe default if resize still isn't provided.
        if resize is None:
            resize = 256

        transform_list = [
            T.Resize(resize),
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