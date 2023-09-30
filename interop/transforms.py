"""Function for transforming paths to images."""
from io import BytesIO
from typing import Callable, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225
ReadImage = transforms.Lambda(
    lambda path: Image.open(
        BytesIO(requests.get(path).content)
        if urlparse(path).scheme.lower() in ("http", "https")
        else path
    )
)


def transform(
    image_size: np.ndarray | int | tuple[int, int],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable[[Sequence[str],], torch.Tensor]:
    """Create a tensor from image paths."""
    if isinstance(image_size, int):
        image_size = image_size, image_size
    image_size = np.asarray(image_size)
    if len(mean) != len(std) != 3:
        raise ValueError("mean and std must have RGB components.")
    img_transforms = transforms.Compose(
        (
            ReadImage,
            transforms.Resize(
                np.ceil(image_size / 0.95).astype(int).tolist(),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        )
    )
    return lambda paths: torch.Tensor(
        np.asarray([img_transforms(path) for path in paths])
    )
