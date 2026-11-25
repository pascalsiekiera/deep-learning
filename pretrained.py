"""Part 4.2: pretrained image encoders (ResNet-50 trained on ImageNet from torchvision, and the
image encoder of CLIP ViT-B/32 from Hugging Face transformers), used frozen in our captioners.

Both models expect 224 x 224 images normalized with their own statistics, so our 64 x 64 images
(normalized with our training statistics by the provided loaders) are mapped back to [0, 1],
upsampled bicubically to 224 x 224 and normalized again (`PretrainedBackbone.prepare`).

A frozen encoder in eval mode (no augmentation) maps every image to fixed features, so we compute
them once for all 8,500 images (`build_feature_store`, saved in checkpoints/q4_2/) and look them
up during training and decoding (`CachedEncoder`). This gives exactly the features the backbone
would compute in every step, at a fraction of the cost. An image is recognized by the values of
32 fixed pixel positions (a fingerprint; all 8,500 fingerprints are checked to be unique).

    encoder = cached_encoder("clip")        # .features(images) (B, 512), .grid(images) (B, 49, 768)
"""

import os
import time
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from common import CHECKPOINT_DIR, get_data

BACKBONES = {
    # name: (pooled dim, grid size, grid dim, normalization mean, std)
    "resnet50": (2048, 7, 2048, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": (512, 7, 768, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}
STORE_DIR = os.path.join(CHECKPOINT_DIR, "q4_2")
SPLITS = ("train", "validation", "test")
N_KEY_PIXELS = 32


class PretrainedBackbone(nn.Module):
    """The pretrained network itself. forward(images) -> (pooled (B, D), grid (B, R, C)).

    resnet50: pooled = global average pooling of the last stage (2048-d), grid = the 7 x 7 map of
        the last stage (2048 channels); ImageNet-1k weights (IMAGENET1K_V2).
    clip: pooled = CLIP's image embedding (512-d, the projected [CLS] token, i.e. the vector CLIP
        aligns with text), grid = the 7 x 7 patch tokens of the last layer (768-d, after the final
        LayerNorm); weights openai/clip-vit-base-patch32."""

    def __init__(self, name, our_mean, our_std, size=224):
        super().__init__()
        self.name, self.size = name, size
        _, _, _, mean, std = BACKBONES[name]
        self.register_buffer("our_mean", torch.as_tensor(our_mean).view(1, 3, 1, 1))
        self.register_buffer("our_std", torch.as_tensor(our_std).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        if name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.body = nn.Sequential(*list(m.children())[:-2])      # without avgpool and fc
        elif name == "clip":
            from transformers import CLIPVisionModelWithProjection
            self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise KeyError(name)
        self.eval()

    def prepare(self, images):
        """Our normalized 64 x 64 images -> the backbone's normalized 224 x 224 input."""
        raw = images * self.our_std + self.our_mean
        raw = F.interpolate(raw, size=self.size, mode="bicubic", align_corners=False).clamp(0, 1)
        return (raw - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images):
        x = self.prepare(images)
        if self.name == "resnet50":
            g = self.body(x)                                         # (B, 2048, 7, 7)
            return g.mean(dim=(2, 3)), g.flatten(2).transpose(1, 2)
        out = self.clip(pixel_values=x)
        tokens = self.clip.vision_model.post_layernorm(out.last_hidden_state[:, 1:])
        return out.image_embeds, tokens                              # (B, 512), (B, 49, 768)


def key_positions():
    g = torch.Generator().manual_seed(0)
    return torch.randperm(64 * 64, generator=g)[:N_KEY_PIXELS]


def fingerprints(images, positions):
    """(B, 3 * N_KEY_PIXELS) pixel values at fixed positions, as tuples (exact float values)."""
    flat = images.flatten(2)[:, :, positions.to(images.device)]
    return [tuple(row) for row in flat.reshape(len(images), -1).cpu().tolist()]


def store_path(name):
    return os.path.join(STORE_DIR, f"features_{name}.pt")


def build_feature_store(name, device=None, batch_size=50):
    """Compute (or load) the pooled and grid features of all images of all splits."""
    path = store_path(name)
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    from captioning import device_for_captioning, image_loader, norm_stats
    device = device or device_for_captioning()
    mean, std = norm_stats()
    backbone = PretrainedBackbone(name, mean, std).to(device)
    positions = key_positions()
    pooled, grid, keys, images_all = [], [], [], []
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for split in SPLITS:
        for x, _ in image_loader(split, batch_size=batch_size):
            p, g = backbone(x.to(device))
            pooled.append(p.float().cpu())
            grid.append(g.half().cpu())
            keys.append(x.flatten(2)[:, :, positions].reshape(len(x), -1))
            images_all.append(x)
    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.time() - t0
    store = {"name": name, "pooled": torch.cat(pooled), "grid": torch.cat(grid),
             "keys": torch.cat(keys), "positions": positions, "seconds": seconds,
             "num_images": sum(len(get_data()[s]["labels"]) for s in SPLITS),
             "split_sizes": [len(get_data()[s]["labels"]) for s in SPLITS],
             "backbone_parameters": sum(p.numel() for p in backbone.parameters())}
    # Every fingerprint must identify one image (identical images may share a fingerprint).
    images_all = torch.cat(images_all)
    seen = {}
    for i, k in enumerate(map(tuple, store["keys"].tolist())):
        if k in seen and not torch.equal(images_all[seen[k]], images_all[i]):
            raise RuntimeError(f"fingerprint collision between images {seen[k]} and {i}")
        seen.setdefault(k, i)
    os.makedirs(STORE_DIR, exist_ok=True)
    torch.save(store, path)
    print(f"saved {path} ({seconds:.0f} s for {store['num_images']} images)")
    return store


class CachedEncoder(nn.Module):
    """A frozen pretrained encoder whose features are looked up in a feature store. Has no
    parameters (the decoder is trained on top of it); the store stays on the CPU and only the
    features of the current batch are moved to the device."""

    def __init__(self, store):
        super().__init__()
        self.name = store["name"]
        self.pooled, self.grid_features = store["pooled"], store["grid"]
        self.positions = store["positions"]
        self.index = {}
        for i, k in enumerate(map(tuple, store["keys"].tolist())):
            self.index.setdefault(k, i)
        self.feature_dim = self.pooled.shape[1]
        self.grid_dim = self.grid_features.shape[2]
        self.grid_size = int(round(self.grid_features.shape[1] ** 0.5))
        self.backbone_parameters = store["backbone_parameters"]

    def lookup(self, images):
        return torch.tensor([self.index[k] for k in fingerprints(images, self.positions)])

    def features(self, images):
        return self.pooled[self.lookup(images)].to(images.device)

    def grid(self, images):
        return self.grid_features[self.lookup(images)].to(images.device, torch.float32)


@lru_cache(maxsize=1)
def cached_encoder(name):
    return CachedEncoder(build_feature_store(name))
