"""
Transforms
----------
Custom transform wrappers that handle both image and target dict
(which contains 'boxes' that must be transformed consistently with the image).

All transforms accept (PIL.Image, target_dict | None) and return the same.
At inference time, target is None and is passed through unchanged.
"""

import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to FloatTensor in [0, 1]."""
    def __call__(self, image, target):
        return TF.to_tensor(image), target


class Normalize:
    """ImageNet normalisation."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, self.mean, self.std)
        return image, target


class RandomHorizontalFlip:
    """
    Flip image and adjust bounding boxes accordingly.
    Only applied during training (target is not None).
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.hflip(image)
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                w = image.width if hasattr(image, "width") else image.shape[-1]
                boxes = target["boxes"].clone()
                # Flip: new_x1 = W - old_x2,  new_x2 = W - old_x1
                boxes[:, 0], boxes[:, 2] = w - target["boxes"][:, 2], w - target["boxes"][:, 0]
                target["boxes"] = boxes
        return image, target


class RandomPhotometricDistort:
    """
    Light colour jitter: brightness, contrast, saturation, hue.
    Boxes are unaffected.
    """
    def __init__(self, p: float = 0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05
        )

    def __call__(self, image, target):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, target


class Resize:
    """
    Resize image so that its shorter side == min_size, keeping aspect ratio,
    but ensuring the longer side does not exceed max_size.
    Boxes are scaled proportionally.
    NOTE: torchvision's Faster R-CNN already does this internally via
    GeneralizedRCNNTransform, so this is mainly for query crops.
    """
    def __init__(self, min_size: int = 900, max_size: int = 1500):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        import PIL.Image as PILImage
        w, h = image.size if isinstance(image, PILImage.Image) else (image.shape[-1], image.shape[-2])

        scale = self.min_size / min(h, w)
        if scale * max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)

        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = TF.resize(image, [new_h, new_w])

        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            target["boxes"] = target["boxes"] * scale

        return image, target
    
class RandomErasingScene:
    def __init__(self, p=0.5, sl=0.02, sh=0.2, r1=0.3, r2=3.3):
        self.p, self.sl, self.sh, self.r1, self.r2 = p, sl, sh, r1, r2

    def __call__(self, image, target):
        import random, math
        if random.random() > self.p or target is None:
            return image, target
        W, H = image.size
        for _ in range(10):
            area = W*H
            target_area = random.uniform(self.sl, self.sh) * area
            aspect = random.uniform(self.r1, self.r2)
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if w < W and h < H:
                x = random.randint(0, W - w); y = random.randint(0, H - h)
                # fill with ImageNet mean (in uint8 space)
                from PIL import ImageDraw
                ImageDraw.Draw(image).rectangle([x, y, x+w, y+h], fill=(124,116,104))
                return image, target
        return image, target


# --------------------------------------------------------------------------- #
#  Ready-made pipelines
# --------------------------------------------------------------------------- #

def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomPhotometricDistort(p=0.5),
        RandomErasingScene(p=0.2),
        ToTensor(),
    ])


def get_eval_transforms():
    return Compose([
        ToTensor(),
    ])
