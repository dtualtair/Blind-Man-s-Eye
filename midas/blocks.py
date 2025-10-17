import torch
import torch.nn as nn

# Import backbone factories and forward helpers for different transformer/CNN encoders
from .backbones.beit import (
    _make_pretrained_beitl16_512,
    _make_pretrained_beitl16_384,
    _make_pretrained_beitb16_384,
    forward_beit,
)
from .backbones.swin_common import (
    forward_swin,
)
from .backbones.swin2 import (
    _make_pretrained_swin2l24_384,
    _make_pretrained_swin2b24_384,
    _make_pretrained_swin2t16_256,
)
from .backbones.swin import (
    _make_pretrained_swinl12_384,
)
from .backbones.levit import (
    _make_pretrained_levit_384,
    forward_levit,
)
from .backbones.vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)


def _make_encoder(
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    in_features=[96, 256, 512, 1024],
):
    """
    Build an encoder composed of:
    - A pretrained feature extractor ("pretrained") built from the selected backbone family.
    - A "scratch" module that adapts backbone feature dimensions to a unified channel size.

    Args:
        backbone (str): Backbone key (e.g., 'beitl16_512', 'swin2l24_384', 'vitb_rn50_384', ...)
        features (int): Target base channel width for the scratch adapter layers.
        use_pretrained (bool): Whether to load pretrained weights for the backbone.
        groups (int): Grouped conv groups used in the scratch convs (default 1).
        expand (bool): If True, progressively increases channel width for deeper layers (x2/x4/x8).
        exportable (bool): Some backbones (e.g., efficientnet) accept this flag for ONNX export friendliness.
        hooks (list[int]): Indices/layer names used to hook intermediate features from the backbone.
        use_vit_only (bool): For 'vitb_rn50_384', use only ViT part (ignore RN50) if True.
        use_readout (str): How to use transformer readout tokens ('ignore', 'add', etc.) depending on backbone code.
        in_features (list[int]): Manual input channel sizes for custom backbones (e.g., Next-ViT).

    Returns:
        (nn.Module, nn.Module): (pretrained_feature_extractor, scratch_adapter)

    Notes:
        - The 'pretrained' module provides .layer{1..4} outputs.
        - The 'scratch' module provides 3 or 4 conv adapters (layer{1..4}_rn) to remap channels to 'features' (or expanded).
    """
    if backbone == "beitl16_512":
        # BEiT-Large with 512 resolution transform settings
        pretrained = _make_pretrained_beitl16_512(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        # Channels from BEiT-Large feature pyramid -> map to target 'features'
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)

    elif backbone == "beitl16_384":
        # BEiT-Large with 384 resolution
        pretrained = _make_pretrained_beitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)

    elif backbone == "beitb16_384":
        # BEiT-Base with 384 resolution
        pretrained = _make_pretrained_beitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)

    elif backbone == "swin2l24_384":
        # SwinV2 Large
        pretrained = _make_pretrained_swin2l24_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([192, 384, 768, 1536], features, groups=groups, expand=expand)

    elif backbone == "swin2b24_384":
        # SwinV2 Base
        pretrained = _make_pretrained_swin2b24_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([128, 256, 512, 1024], features, groups=groups, expand=expand)

    elif backbone == "swin2t16_256":
        # SwinV2 Tiny
        pretrained = _make_pretrained_swin2t16_256(use_pretrained, hooks=hooks)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)

    elif backbone == "swinl12_384":
        # Swin Large (12)
        pretrained = _make_pretrained_swinl12_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([192, 384, 768, 1536], features, groups=groups, expand=expand)

    elif backbone == "next_vit_large_6m":
        # Next-ViT Large (custom in_features)
        from .backbones.next_vit import _make_pretrained_next_vit_large_6m

        pretrained = _make_pretrained_next_vit_large_6m(hooks=hooks)
        scratch = _make_scratch(in_features, features, groups=groups, expand=expand)

    elif backbone == "levit_384":
        # LeViT compact backbone
        pretrained = _make_pretrained_levit_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([384, 512, 768], features, groups=groups, expand=expand)

    elif backbone == "vitl16_384":
        # ViT-L/16
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)

    elif backbone == "vitb_rn50_384":
        # ViT-B/16 + ResNet-50 hybrid
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained, hooks=hooks, use_vit_only=use_vit_only, use_readout=use_readout
        )
        scratch = _make_scratch([256, 512, 768, 768], features, groups=groups, expand=expand)

    elif backbone == "vitb16_384":
        # ViT-B/16
        pretrained = _make_pretrained_vitb16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)

    elif backbone == "resnext101_wsl":
        # ResNeXt-101 WSL pretrained
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)

    elif backbone == "efficientnet_lite3":
        # EfficientNet-Lite3 backbone
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)

    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """
    Create a 'scratch' module to adapt multi-scale backbone features to a common channel width.

    Args:
        in_shape (list[int]): Input channels for each backbone stage [l1, l2, l3, (l4)]
        out_shape (int): Base output channels for stage 1 (may expand for deeper stages)
        groups (int): Grouped convolution setting for all adapter convs
        expand (bool): If True, set stage channels progressively as:
                       out1=out_shape, out2=out_shape*2, out3=out_shape*4, out4=out_shape*8 (if present)

    Returns:
        nn.Module: Container with conv adapters layer{1..4}_rn to remap channels.
    """
    scratch = nn.Module()

    # Default: same out_shape for all
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    # Progressive expansion if requested
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    # Stage 1 adapter
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # Stage 2 adapter
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # Stage 3 adapter
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # Stage 4 adapter (if present)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    """
    Fetch EfficientNet-Lite3 via torch.hub and convert it to a 4-stage backbone
    with .layer1..layer4 feature outputs.
    """
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    """
    Adapt a timm-style EfficientNet model into a simple module with:
    layer1, layer2, layer3, layer4 sequential feature blocks.
    """
    pretrained = nn.Module()

    # stem + first blocks (tuned to align with MiDaS feature pyramid expectations)
    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_resnet_backbone(resnet):
    """
    Convert a torchvision-like ResNet into a backbone with layer1..layer4 outputs
    compatible with the rest of the MiDaS/DPT pipeline.
    """
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    """
    Load ResNeXt-101 (32x8d) pre-trained on weakly-supervised learning (WSL) dataset
    via torch.hub and return a standardized backbone.
    """
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


class Interpolate(nn.Module):
    """A small wrapper module for nn.functional.interpolate to use inside nn.Sequential graphs."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """
        Args:
            scale_factor (float): Spatial scaling factor for H and W.
            mode (str): Interpolation mode ('nearest', 'bilinear', 'bicubic', ...).
            align_corners (bool): See PyTorch docs; relevant for 'bilinear'/'bicubic'.
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Apply interpolation to input tensor."""
        return self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )


class ResidualConvUnit(nn.Module):
    """
    Residual convolutional unit used in DPT/MiDaS decoders.

    Structure:
        x -> ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> (+ x) -> out
    """

    def __init__(self, features: int):
        """
        Args:
            features (int): Number of input/output channels (RCU is channel-preserving).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x  # Residual addition


class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block that merges features from different scales and upsamples.

    Typical usage:
        y = FFB(x)           # single input: refine and upsample
        y = FFB(x_high, x_low)  # fuse lower-res 'x_high' with skip 'x_low', then upsample
    """

    def __init__(self, features: int):
        """
        Args:
            features (int): Number of channels expected on inputs and outputs.
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            xs: Either (x,) or (x, skip). If two tensors provided, they are fused.

        Returns:
            torch.Tensor: Refined and upsampled feature map.
        """
        output = xs[0]

        # If a second feature map is provided, refine and add it
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        # Further refinement
        output = self.resConfUnit2(output)

        # Upsample by factor 2 (bilinear)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class ResidualConvUnit_custom(nn.Module):
    """
    Residual convolutional unit with optional BatchNorm and quantization-friendly add.

    Structure:
        x -> act -> Conv -> (BN) -> act -> Conv -> (BN) -> (+ x) -> out
    """

    def __init__(self, features: int, activation: nn.Module, bn: bool):
        """
        Args:
            features (int): Channel dimension for in/out.
            activation (nn.Module): Activation function to use (e.g., nn.ReLU()).
            bn (bool): Whether to use BatchNorm2d after each conv.
        """
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        # Quantization-friendly residual add (avoids issues with + in quantized graphs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional BN and quantized-safe residual add."""
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        # If grouped convs > 1, optionally merge here (placeholder for extensions)
        if self.groups > 1:
            out = self.conv_merge(out)

        # Residual addition via FloatFunctional for quantization compatibility
        return self.skip_add.add(out, x)
        # return out + x  # Standard residual add (non-quantized)


class FeatureFusionBlock_custom(nn.Module):
    """
    Custom feature fusion block with:
    - Two ResidualConvUnit_custom refinements
    - Optional channel reduction (expand=False halves channels at the end)
    - Flexible upsampling: scale by 2 or to a specified spatial size
    - Quantization-friendly residual addition

    Typical usage:
        y = FFB_custom(x)
        y = FFB_custom(x_high, x_low)
        y = FFB_custom(x, size=(H, W))  # upsample to target size
    """

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size=None,
    ):
        """
        Args:
            features (int): Channel width for internal processing.
            activation (nn.Module): Activation function to use within RCUs.
            deconv (bool): Placeholder for transpose-conv upsampling (not used here).
            bn (bool): Use BatchNorm in RCUs if True.
            expand (bool): If True, reduce channels by 2 at the output (features -> features//2).
            align_corners (bool): Interpolate align_corners setting for bilinear upsample.
            size (tuple[int,int] or None): If set, upsample to this spatial size; else scale by 2.
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1

        self.expand = expand
        out_features = features if not self.expand else features // 2

        # 1x1 conv to set final output channels (optionally reduced)
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        # Two refinement RCUs
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        # Quantization-friendly add
        self.skip_add = nn.quantized.FloatFunctional()

        # Optional fixed output size for upsample
        self.size = size

    def forward(self, *xs: torch.Tensor, size=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            xs: One or two tensors to fuse. If two, second is a skip connection from shallower layer.
            size (tuple[int,int] or None): Optional override for output spatial size (H, W).

        Returns:
            torch.Tensor: Refined, upsampled, and channel-adjusted feature map.
        """
        output = xs[0]

        # Fuse with skip connection if provided
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)  # quantization-safe addition
            # output += res  # standard add

        # Second refinement
        output = self.resConfUnit2(output)

        # Choose upsampling strategy: scale_factor=2 or to an explicit size
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        # Final channel adjustment
        output = self.out_conv(output)
        return output
