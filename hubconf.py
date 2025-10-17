"""
Hub configuration for MiDaS models used by Blind Man's Eye (BME)
================================================================

This module exposes factory functions to instantiate various MiDaS/DPT models
with optional pretrained weights, so they can be loaded via torch.hub or
directly imported within the BME pipeline.

- Provides wrappers around backbone variants (BEiT, Swin, ViT, LeViT, Next-ViT)
- Downloads official weights from Intel's MiDaS releases when pretrained=True
- Includes a set of input transforms matching each model family

Usage examples:
---------------
# Load a model directly
from hubconf import DPT_Hybrid
model = DPT_Hybrid(pretrained=True)

# Or via torch.hub (from a repo that contains this file):
model = torch.hub.load("Ekveer-Sahoo/Blind-Man-s-Eye", "DPT_Hybrid", pretrained=True)

Notes:
- All models are configured for non-negative depth by default.
- The small model (MiDaS_small) is recommended for real-time applications.
"""

dependencies = ["torch"]

import torch

# Core MiDaS/DPT model classes
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small


def DPT_BEiT_L_512(pretrained: bool = True, **kwargs):
    """MiDaS DPT_BEiT_L_512 model for monocular depth estimation.
    Args:
        pretrained: If True, downloads and loads official pretrained weights.
    Returns:
        torch.nn.Module: DPTDepthModel (BEiT-Large, 512 input transform family)
    """
    model = DPTDepthModel(
        path=None,
        backbone="beitl16_512",  # BEiT Large @ 512
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_BEiT_L_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_BEiT_L_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="beitl16_384",  # BEiT Large @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_BEiT_B_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_BEiT_B_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="beitb16_384",  # BEiT Base @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_SwinV2_L_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_SwinV2_L_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="swin2l24_384",  # SwinV2 Large @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_SwinV2_B_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_SwinV2_B_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="swin2b24_384",  # SwinV2 Base @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_SwinV2_T_256(pretrained: bool = True, **kwargs):
    """MiDaS DPT_SwinV2_T_256 (Tiny) model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="swin2t16_256",  # SwinV2 Tiny @ 256
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_Swin_L_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_Swin_L_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="swinl12_384",  # Swin Large @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_Next_ViT_L_384(pretrained: bool = True, **kwargs):
    """MiDaS DPT_Next_ViT_L_384 model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="next_vit_large_6m",  # Next-ViT Large @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_LeViT_224(pretrained: bool = True, **kwargs):
    """MiDaS DPT_LeViT_224 model for monocular depth estimation (compact)."""
    model = DPTDepthModel(
        path=None,
        backbone="levit_384",      # LeViT backbone (lightweight)
        non_negative=True,
        head_features_1=64,
        head_features_2=8,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_Large(pretrained: bool = True, **kwargs):
    """MiDaS DPT-Large (ViT-L/16) model for monocular depth estimation."""
    model = DPTDepthModel(
        path=None,
        backbone="vitl16_384",  # Vision Transformer Large @ 384
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def DPT_Hybrid(pretrained: bool = True, **kwargs):
    """MiDaS DPT-Hybrid (ViT-B + ResNet-50) model for monocular depth estimation.
    Good trade-off between speed and accuracy.
    """
    model = DPTDepthModel(
        path=None,
        backbone="vitb_rn50_384",  # ViT-Base + RN50 hybrid
        non_negative=True,
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def MiDaS(pretrained: bool = True, **kwargs):
    """MiDaS v2.1 model for monocular depth estimation (original CNN variant)."""
    model = MidasNet()

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def MiDaS_small(pretrained: bool = True, **kwargs):
    """MiDaS v2.1 small model for resource-constrained devices.
    Recommended for real-time BME use-cases (mobile/embedded).
    """
    model = MidasNet_small(
        None,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={"expand": True},
    )

    if pretrained:
        checkpoint = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def transforms():
    """Return a namespace-like object that contains predefined input transform
    pipelines for different MiDaS/DPT model families.

    The transforms handle:
    - Resizing to the expected encoder resolution
    - Aspect ratio handling and multiple-of-32 alignment
    - Normalization to model-appropriate mean/std
    - Preparing for network (CHW tensor) and adding batch dimension
    """
    import cv2
    from torchvision.transforms import Compose
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from midas import transforms as _tx  # reuse the module as a container

    # Default transform (upper_bound resize keeps aspect ratio)
    _tx.default_transform = Compose(
        [
            lambda img: {"image": img / 255.0},  # to [0,1]
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            # Convert dict["image"] ndarray(HWC) -> torch tensor (1, C, H, W)
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # Small model (256) transform for MiDaS_small
    _tx.small_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # DPT family (minimal resize, 384) with symmetric mean/std
    _tx.dpt_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",  # preserve as much as possible
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # BEiT-512 variant
    _tx.beit512_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                512,
                512,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # Swin-384 (aspect ratio not preserved)
    _tx.swin384_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=False,  # required by some Swin configs
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # Swin-256 (aspect ratio not preserved)
    _tx.swin256_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # LeViT-224 (square, no aspect ratio preservation)
    _tx.levit_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                224,
                224,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    # Return the module with attributes set as a transform namespace
    return _tx
