from logging import warning
from typing import cast

import torch
import torch.nn as nn
import monai.networks.nets as nets

from timm.models.layers import DropPath, Mlp
from .vit import ViT, load_pretrained


class Multi2Di(nn.Module):
    """
    Multi-2D input model based on MONAI's MIL model. Input is a 5D tensor (B, N, C, H, W) where B is the batch size,
    N is the number of slices, C is the number of channels, H is the height, and W is the width.

    Args:
        in_channels: number of input channels.
        num_classes: number of output classes.
        mil_mode: MIL algorithm, available values (Defaults to ``"att"``):

            - ``"att"`` - attention based MIL https://arxiv.org/abs/1802.04712.
            - ``"att_trans"`` - transformer MIL https://arxiv.org/abs/2111.01556.
            - ``"att_trans_pyramid"`` - transformer pyramid MIL https://arxiv.org/abs/2111.01556.

        img_size: input image size.
        backbone: Backbone classifier CNN (either ``None``, a ``nn.Module`` that returns features,
            or a string name of a torchvision model).
        backbone_num_features: Number of output features of the backbone CNN
            Defaults to ``None`` (necessary only when using a custom backbone)
        trans_blocks: number of the blocks in `TransformEncoder` layer.
        trans_dropout: dropout rate in `TransformEncoder` layer.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = None,
        use_auto_att: bool = False,
        feat_comb_mode: str = "att",
        backbone: str | nn.Module | None = None,
        backbone_chkp: str | None = None,
        backbone_num_features: int | None = None,
        trans_blocks: int = 4,
        trans_dropout: float = 0.1,
        use_mil_head: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive: " + str(num_classes))

        if feat_comb_mode.lower() not in ["att", "att_trans", "att_trans_pyramid"]:
            raise ValueError("Unsupported mil_mode: " + str(feat_comb_mode))
        
        if backbone.lower() not in ["resnet50", "retfound"]:
            raise ValueError("Unsupported backbone: " + str(backbone))

        self.feat_comb_mode = feat_comb_mode.lower()
        self.use_auto_att = use_auto_att
        self.use_mil_head = use_mil_head
        self.attention = nn.Sequential()
        self.transformer: nn.Module | None = None

        resnet_params = {
            # model_name: (block, layers, shortcut_type, bias_downsample)
            "resnet10": ("basic", [1, 1, 1, 1], "B", False),
            "resnet18": ("basic", [2, 2, 2, 2], "A", True),
            "resnet34": ("basic", [3, 4, 6, 3], "A", True),
            "resnet50": ("bottleneck", [3, 4, 6, 3], "B", False),
            "resnet101": ("bottleneck", [3, 4, 23, 3], "B", False),
            "resnet152": ("bottleneck", [3, 8, 36, 3], "B", False),
            "resnet200": ("bottleneck", [3, 24, 36, 3], "B", False),
        }

        if backbone == "resnet50":
            block, layers, shortcut_type, bias_downsample = resnet_params["resnet50"]
            net = nets.ResNet(
                block=block,
                layers=layers,
                block_inplanes=[64, 128, 256, 512],
                shortcut_type=shortcut_type,
                bias_downsample=bias_downsample,
                num_classes=num_classes,
                n_input_channels=in_channels,
                spatial_dims=2,
            )
            nfc = net.fc.in_features  # save the number of final features
            net.fc = torch.nn.Identity()  # remove final linear layer

            self.extra_outputs: dict[str, torch.Tensor] = {}

            if feat_comb_mode == "att_trans_pyramid":
                # register hooks to capture outputs of intermediate layers
                def forward_hook(layer_name):

                    def hook(module, input, output):
                        self.extra_outputs[layer_name] = output

                    return hook

                net.layer1.register_forward_hook(forward_hook("layer1"))
                net.layer2.register_forward_hook(forward_hook("layer2"))
                net.layer3.register_forward_hook(forward_hook("layer3"))
                net.layer4.register_forward_hook(forward_hook("layer4"))

        elif backbone == "retfound":
            net = ViT(img_size=img_size, patch_size=16, embed_dim=1024, in_chans=in_channels,
                      num_classes=num_classes, depth=24, num_heads=16, mlp_ratio=4)
            nfc = 1024
            if backbone_chkp is not None:
                load_pretrained(backbone_chkp, net)
            else:
                warning("No pretrained weights loaded for ViT model")

            # Remove the classification head
            net.head = nn.Identity()

        elif isinstance(backbone, nn.Module):
            # use a custom backbone
            net = backbone
            nfc = backbone_num_features

            if backbone_num_features is None:
                raise ValueError("Number of endencoder features must be provided for a custom backbone model")
        else:
            raise ValueError("Unsupported backbone")

        if backbone != "resnet50" and feat_comb_mode == "att_trans_pyramid":
            raise ValueError("Only resnet50 is supported for the mode:" + str(feat_comb_mode))

        if self.use_auto_att:
            self.auto_att = AutoAttentionEncoder(embed_dim=nfc, num_heads=8, n_layers=1)

        if self.feat_comb_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, nfc), nn.Tanh(), nn.Linear(nfc, 1))

        elif self.feat_comb_mode == "att_trans":
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.feat_comb_mode == "att_trans_pyramid":
            transformer_list = nn.ModuleList(
                [
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout), num_layers=trans_blocks
                    ),
                    nn.Sequential(
                        nn.Linear(768, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=2304, nhead=8, dropout=trans_dropout),
                        num_layers=trans_blocks,
                    ),
                ]
            )
            self.transformer = transformer_list
            nfc = nfc + 256
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        else:
            raise ValueError("Unsupported feature combination mode: " + str(feat_comb_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.net = net

    def calc_head(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape

        if self.feat_comb_mode == "att":
            a = self.attention(x)
            a = torch.softmax(a, dim=1)

        elif self.feat_comb_mode == "att_trans" and self.transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)

        elif self.feat_comb_mode == "att_trans_pyramid" and self.transformer is not None:
            l1 = torch.mean(self.extra_outputs["layer1"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l2 = torch.mean(self.extra_outputs["layer2"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l3 = torch.mean(self.extra_outputs["layer3"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l4 = torch.mean(self.extra_outputs["layer4"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)

            transformer_list = cast(nn.ModuleList, self.transformer)

            x = transformer_list[0](l1)
            x = transformer_list[1](torch.cat((x, l2), dim=2))
            x = transformer_list[2](torch.cat((x, l3), dim=2))
            x = transformer_list[3](torch.cat((x, l4), dim=2))

            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)

        elif self.feat_comb_mode == "identity":
            a = torch.ones(sh[1], 1).to(x.device)

        else:
            raise ValueError("Wrong model mode" + str(self.feat_comb_mode))

        x = torch.sum(x * a, dim=1) if self.use_mil_head else x * a
        x = self.myfc(x)
        
        return x, a

    def forward(self, x: torch.Tensor, return_att: bool = False) -> torch.Tensor:
        sh = x.shape

        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        x = self.net(x)

        x = x.reshape(sh[0], sh[1], -1)

        if self.use_auto_att:
            x = self.auto_att(x)

        x, att = self.calc_head(x) # Without MIL: (B, N, C). With MIL: (B, C).

        if not self.use_mil_head:
            x = x.reshape(sh[0] * sh[1], -1) # (B, N, C) -> (B * N, C)

        if return_att:
            return x, att
        else:
            return x


class AutoAttentionLayer(nn.Module):
    """
    AutoAttentionLayer module that applies multi-head attention mechanism on the input tensor.
        embed_dim (int): The dimension of the input tensor.
        mlp_ratio (int, optional): The ratio of the hidden dimension to the input dimension in the MLP layer. Default is 2.
        num_heads (int, optional): The number of attention heads. Default is 1.
        drop_path (float, optional): The dropout probability for the drop path layers. Default is 0.1.
        Forward pass of the AutoAttentionLayer module.
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying multi-head attention and MLP layers.
    """
    
    def __init__(self, embed_dim, num_heads=1, mlp_ratio=2, drop_path=0.1):
        super().__init__()
        self.auto_att = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_path)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.mlp = Mlp(embed_dim,
                       embed_dim * mlp_ratio,
                       embed_dim,
                       drop=0.1)
        nn.init.xavier_uniform_(self.mlp.fc1.weight)
        nn.init.xavier_uniform_(self.mlp.fc2.weight)
        nn.init.constant_(self.mlp.fc1.bias, 0)
        nn.init.constant_(self.mlp.fc2.bias, 0)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.drop_path1(self.auto_att(x, x, x)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class AutoAttentionEncoder(nn.Module):
    """
    AutoAttentionEncoder module that applies multiple AutoAttentionLayer to the input tensor.
    Args:
        embed_dim (int): The dimension of the input tensor.
        num_heads (int, optional): The number of attention heads. Defaults to 1.
        n_layers (int, optional): The number of AutoAttentionLayer layers to apply. Defaults to 1.
    Returns:
        torch.Tensor: The output tensor after applying multiple AutoAttentionLayer layers.
    """
    
    
    def __init__(self, embed_dim, num_heads=1, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([AutoAttentionLayer(embed_dim, num_heads=num_heads)
                                     for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
