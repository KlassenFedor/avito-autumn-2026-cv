"""MobileNetV3-Small с одним логитом; одна и та же архитектура при обучении и инференсе."""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def make_model(initialization="random", weights_path=None):
    # веса ImageNet скачиваются только при инициализации перед обучением
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if initialization == "imagenet" and not weights_path else None
    model = mobilenet_v3_small(weights=weights)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)  # исходный state_dict torchvision на 1000 классов
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def save_checkpoint(path, model, height, width, temperature=1.0, **extra):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "height": height, "width": width, "temperature": temperature,
                "architecture": "mobilenet_v3_small", **extra}, path)


@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        # оценка в FP32 независимо от того, обучалась ли модель с AMP
        chunks.append(model(images).squeeze(1).float().cpu())
    if not chunks:
        raise ValueError("Пустой загрузчик для оценки")
    return torch.cat(chunks).numpy()
