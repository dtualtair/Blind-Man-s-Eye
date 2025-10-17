import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features: int, use_bn: bool, size=None) -> FeatureFusionBlock_custom:
    """
    Helper to build a feature-fusion block used in the decoder/refinement path.

    Args:
        features: channel width used inside the fusion block
        use_bn: enable/disable BatchNorm in the residual units
        size: optional spatial size to upsample to (H, W), otherwise use scale_factor=2

    Returns:
        FeatureFusionBlock_custom: a configured fusion block
    """
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),     # activation function
        deconv=False,       # not using transpose conv here
        bn=use_bn,          # optional BatchNorm inside RCUs
        expand=False,       # keep same channels through the block
        align_corners=True, # bilinear upsample align_corners
        size=size,          # fixed output size if provided
    )


class DPT(BaseModel):
    """
    Dense Prediction Transformer (DPT) meta-architecture.

    This class wires:
      - a chosen transformer/CNN backbone encoder producing multi-scale features
      - a "scratch" adapter to align feature channels
      - a refinement/decoder that fuses features and upsamples to prediction resolution
      - a "head" module (passed in ctor) to produce the final per-pixel output

    The design supports multiple backbones (BEiT, Swin v1/v2, ViT, LeViT, Next-ViT).
    """

    def __init__(
        self,
        head: nn.Module,
        features: int = 256,
        backbone: str = "vitb_rn50_384",
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        **kwargs
    ):
        """
        Args:
            head: output head module producing final prediction from last refined feature
            features: base channel width for decoder/refiner 'scratch' modules
            backbone: encoder backbone key (e.g., 'vitb_rn50_384', 'swin2b24_384', ...)
            readout: how to use transformer readout tokens ('project', 'ignore', ...)
            channels_last: set contiguous memory_format to channels_last during forward if True
            use_bn: use BatchNorm in residual units of fusion blocks
            **kwargs: forwarded to _make_encoder / backbone builders
        """
        super(DPT, self).__init__()

        self.channels_last = channels_last

        # Hook indices per backbone define which intermediate features are extracted.
        # Some hierarchical backbones (Swin/LeViT/Next-ViT) have restricted valid ranges.
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],   # Allowed ranges: [0,1], [0,1], [0,17], [0,1]
            "swin2b24_384": [1, 1, 17, 1],   # Same as above
            "swin2t16_256": [1, 1, 5, 1],    # [0,1], [0,1], [0,5], [0,1]
            "swinl12_384": [1, 1, 17, 1],    # [0,1], [0,1], [0,17], [0,1]
            "next_vit_large_6m": [2, 6, 36, 39],  # [0,2], [3,6], [7,36], [37,39]
            "levit_384": [3, 11, 21],        # [0,3], [6,11], [14,21]  (3 outputs)
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384":   [2, 5, 8, 11],
            "vitl16_384":   [5, 11, 17, 23],
        }[backbone]

        # Next-ViT provides non-standard channel dimensions; pass them explicitly.
        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Build encoder (pretrained features + scratch adapters).
        # Note: 'use_pretrained' set False here; switch to True to load ImageNet weights when training.
        self.pretrained, self.scratch = _make_encoder(
            backbone=backbone,
            features=features,
            use_pretrained=False,  # set True to initialize from ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        # Number of feature maps produced by the backbone (3 for LeViT, 4 otherwise)
        self.number_layers = len(hooks) if hooks is not None else 4

        # For LeViT, the refinenet3 upsamples to a fixed size; others use scale/size from inputs
        size_refinenet3 = None
        self.scratch.stem_transpose = None  # Optional final transpose/stem for LeViT

        # Select appropriate forward function based on backbone family
        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        elif "next_vit" in backbone:
            from .backbones.next_vit import forward_next_vit
            self.forward_transformer = forward_next_vit
        elif "levit" in backbone:
            self.forward_transformer = forward_levit
            size_refinenet3 = 7  # specific upsample target for LeViT at this stage
            # additional stem to adjust channels/resolution post decoder for LeViT
            self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        else:
            self.forward_transformer = forward_vit

        # Build refinement/decoder path: 3 or 4 fusion blocks depending on #layers
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Final output head (e.g., conv -> upsample -> conv -> relu -> conv -> relu/identity)
        self.scratch.output_conv = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder -> scratch adapters -> refinement decoder -> head.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            torch.Tensor: Final prediction tensor (B, C_out, H_out, W_out)
        """
        # Optionally switch to channels_last for performance on some hardware
        if self.channels_last is True:
            x.contiguous(memory_format=torch.channels_last)

        # Extract multi-scale features from the backbone
        layers = self.forward_transformer(self.pretrained, x)

        # Unpack features based on the number of outputs
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        # Align channels via scratch adapters
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Top-down fusion/upsampling path
        if self.number_layers == 3:
            # For 3-stage backbones (e.g., LeViT): start from third stage
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            # For 4-stage backbones: fuse stage 4 into 3, then continue
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])

        # Fuse with stage 2, then with stage 1
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Optional transpose/stem for LeViT variant
        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        # Produce final per-pixel prediction
        out = self.scratch.output_conv(path_1)
        return out


class DPTDepthModel(DPT):
    """
    Concrete DPT model for depth estimation.

    Adds a default "head" that outputs a single-channel depth (with optional non-negativity).
    """

    def __init__(self, path: str = None, non_negative: bool = True, **kwargs):
        """
        Args:
            path: Optional checkpoint path for loading pre-trained weights
            non_negative: If True, applies ReLU on output to enforce non-negative depth
            **kwargs: forwarded to base DPT (e.g., backbone, features, readout, etc.)
        """
        # Configure head channel widths
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        # Remove from kwargs after consuming
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        # Decoder head:
        #   conv (C -> C/2) -> upsample x2 -> conv (C/2 -> 32) -> ReLU
        #   -> conv (32 -> 1) -> ReLU (optional non-negative) -> Identity
        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),  # placeholder to keep structure similar to reference
        )

        # Initialize base DPT with the constructed head
        super().__init__(head, **kwargs)

        # Optionally load weights from path (supports raw state_dict or dict with "model")
        if path is not None:
            self.load(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a single-channel depth map.

        Returns:
            (B, H, W) tensor after squeezing the channel dimension.
        """
        return super().forward(x).squeeze(dim=1)
