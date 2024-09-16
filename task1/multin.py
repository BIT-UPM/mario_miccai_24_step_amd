import torch
import torch.nn as nn
import monai.networks.nets as nets

from logging import warning
from timm.layers import Mlp, DropPath
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
        num_auto_att_layers: int = 2,
        cross_att_type: str = 'cross',
        bicross_att_reduction: str = 'mean',
        num_cross_att_layers: int = 1,
        final_proj: str = 'linear',
        backbone: str | nn.Module | None = None,
        backbone_chkp: str | None = None,
        backbone_num_features: int | None = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive: " + str(num_classes))
        
        if backbone.lower() not in ["resnet50", "retfound"]:
            raise ValueError("Unsupported backbone: " + str(backbone))
        
        self.net, nfc = self.get_backbone(backbone, backbone_chkp, backbone_num_features, num_classes, in_channels, img_size)
        self.auto_att = AutoAttentionEncoder(nfc, num_heads=4, n_layers=num_auto_att_layers) if use_auto_att else None
        self.layer_norm = nn.LayerNorm(nfc, eps=1e-6)

        if cross_att_type == 'cross':
            self.cross_att = CrossAttentionEncoder(nfc, num_heads=4, n_layers=num_cross_att_layers)
        elif cross_att_type == 'bicross':
            self.cross_att = BiCrossAttentionEncoder(nfc, nfc, nfc, num_cross_att_layers, bicross_att_reduction)
        else:
            raise ValueError("Unsupported cross-attention type: " + str(cross_att_type))

        if cross_att_type == 'bicross' and bicross_att_reduction == 'concat':
            nfc = nfc * 2

        if final_proj == 'linear':
            self.final_proj = nn.Linear(nfc, num_classes)
            self.head = nn.Identity()
        elif final_proj == 'lstm':
            self.final_proj = nn.LSTM(nfc, nfc, num_layers=2, batch_first=True,
                                      dropout=0.1, bidirectional=False)
            nfc = nfc * 2 if self.final_proj.bidirectional else nfc
        elif final_proj == 'rnn':
            self.final_proj = nn.RNN(nfc, nfc, num_layers=1, batch_first=True,
                                     dropout=0.1, bidirectional=True)
            nfc = nfc * 2 if self.final_proj.bidirectional else nfc
        # Initialize all weights of the LSTM using a for loop
        for name, param in self.final_proj.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        if final_proj == 'linear':
            return

        self.head = nn.Linear(nfc, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def get_backbone(self,
                     backbone,
                     backbone_chkp,
                     backbone_num_features,
                     num_classes,
                     in_channels,
                     img_size) -> None:
        
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

        elif backbone == "retfound":
            net = ViT(img_size=img_size, patch_size=16, embed_dim=1024, in_chans=in_channels,
                      num_classes=num_classes, depth=24, num_heads=16, mlp_ratio=4)
            nfc = 1024
            if backbone_chkp is not None:
                load_pretrained(backbone_chkp, net)
            else:
                warning("No pretrained weights loaded for ViT OCT model")

            # Remove the classification head
            net.head = nn.Identity()

            # Freeze the backbone (GPU memory saving)
            for param in net.parameters():
                param.requires_grad = False

            # # Unfreeze the later layers
            for param in net.blocks[:].parameters():
                param.requires_grad = True

        elif isinstance(backbone, nn.Module):
            # use a custom backbone
            net = backbone
            nfc = backbone_num_features

            if backbone_num_features is None:
                raise ValueError("Number of endencoder features must be provided for a custom backbone model")
        else:
            raise ValueError("Unsupported backbone")
        
        return net, nfc

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        sh_0 = x_0.shape
        sh_1 = x_1.shape

        x_0 = x_0.reshape(sh_0[0] * sh_0[1], sh_0[2], sh_0[3], sh_0[4])
        x_1 = x_1.reshape(sh_1[0] * sh_1[1], sh_1[2], sh_1[3], sh_1[4])
        x_0 = self.net(x_0)
        x_1 = self.net(x_1)

        x_0 = x_0.reshape(sh_0[0], sh_0[1], -1)
        x_1 = x_1.reshape(sh_1[0], sh_1[1], -1)

        if self.auto_att is not None:
            x_0 = self.auto_att(x_0)
            x_1 = self.auto_att(x_1)

        out = self.cross_att(x_0, x_1)

        if isinstance(self.final_proj, nn.LSTM):
            out, _ = self.final_proj(out)
        else:
            out = self.final_proj(out)

        out = self.head(out)
        out = out.reshape(sh_0[0] * sh_0[1], -1) # (B, N, C) -> (B * N, C)

        return out


class BiCrossAttention(nn.Module):
    """
    Modified CrossAttention module that performs cross-attention between two input tensors but
    returns the average of the context vectors (values_1, values_2).
    Args:
        d_in (int): The input dimension of the tensors.
        d_out_kq (int): The output dimension of the query and key tensors.
        d_out_v (int): The output dimension of the value tensor.
    Attributes:
        W_query_1 (nn.Parameter): The learnable weight matrix for the query tensor of the first tensor.
        W_key_1 (nn.Parameter): The learnable weight matrix for the key tensor of the first tensor.
        W_value_1 (nn.Parameter): The learnable weight matrix for the value tensor of the first tensor.
        W_query_2 (nn.Parameter): The learnable weight matrix for the query tensor of the second tensor.
        W_key_2 (nn.Parameter): The learnable weight matrix for the key tensor of the second tensor.
        W_value_2 (nn.Parameter): The learnable weight matrix for the value tensor of the second tensor.
    Methods:
        forward(x_1, x_2): Performs the forward pass of the cross-attention module.
        forward_ca(x_1, x_2, W_query, W_key, W_value): Performs the cross-attention operation between two input tensors.
    Returns:
        (torch.Tensor, torch.Tensor): The context vectors obtained from the cross-attention operation.
        
    """

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query_1 = nn.Parameter(torch.empty(d_in, d_out_kq))
        self.W_key_1   = nn.Parameter(torch.empty(d_in, d_out_kq))
        self.W_value_1 = nn.Parameter(torch.empty(d_in, d_out_v))

        self.W_query_2 = nn.Parameter(torch.empty(d_in, d_out_kq))
        self.W_key_2   = nn.Parameter(torch.empty(d_in, d_out_kq))
        self.W_value_2 = nn.Parameter(torch.empty(d_in, d_out_v))

        # Initialize weights
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def forward_ca(self, x_1, x_2, W_query, W_key, W_value):
        queries_1 = x_1 @ W_query

        keys_2 = x_2 @ W_key
        values_2 = x_2 @ W_value

        attn_scores = queries_1 @ torch.transpose(keys_2, 1, 2)
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)

        context_vec = attn_weights @ values_2
        return context_vec
    
    def forward(self, x_1, x_2):
        context_vec_2 = self.forward_ca(x_1, x_2, self.W_query_1, self.W_key_2, self.W_value_2)
        context_vec_1 = self.forward_ca(x_2, x_1, self.W_query_2, self.W_key_1, self.W_value_1)

        return context_vec_1, context_vec_2


class BiCrossAttentionLayer(nn.Module):
    """
    BiCrossAttentionEncoder module that performs cross-attention between two input tensors and
    uses AddAndNorm module to normalize the output. This operation is performed n times.
    Args:
        d_in (int): The input dimension of the tensors.
        d_out_kq (int): The output dimension of the query and key tensors.
        d_out_v (int): The output dimension of the value tensor.
    Methods:
        forward(x_1, x_2): Performs the forward pass of the BiCrossAttentionEncoder module.
    Returns:
        torch.Tensor: The normalized tensor obtained from the BiCrossAttention operation.
    """
    
    def __init__(self, d_in, d_out_kq, d_out_v, mlp_ratio=2, drop_path=0.0):
        super().__init__()
        self.cross_att = BiCrossAttention(d_in, d_out_kq, d_out_v)

        self.norm1 = nn.LayerNorm(d_in, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_in, eps=1e-6)

        self.mlp = Mlp(d_in,
                       d_in * mlp_ratio,
                       d_in,
                       drop=0.1)
        nn.init.xavier_uniform_(self.mlp.fc1.weight)
        nn.init.xavier_uniform_(self.mlp.fc2.weight)
        nn.init.constant_(self.mlp.fc1.bias, 0)
        nn.init.constant_(self.mlp.fc2.bias, 0)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_1, x_2):
        c1, c2 = self.cross_att(self.norm1(x_1), self.norm1(x_2))
        x_1 = x_1 + self.drop_path1(c1)
        x_2 = x_2 + self.drop_path1(c2)
        out1 = x_1 + self.drop_path2(self.mlp(self.norm2(x_1)))
        out2 = x_2 + self.drop_path2(self.mlp(self.norm2(x_2)))

        return out1, out2


class BiCrossAttentionEncoder(nn.Module):
    """
    BiCrossAttentionEncoder module that performs cross-attention between two input tensors and
    uses AddAndNorm module to normalize the output. This operation is performed n times.
    Args:
        d_in (int): The input dimension of the tensors.
        d_out_kq (int): The output dimension of the query and key tensors.
        d_out_v (int): The output dimension of the value tensor.
        n_layers (int): The number of BiCrossAttentionLayer layers to be used.
    Attributes:
        layers (nn.ModuleList): The list of BiCrossAttentionLayer layers.
    Methods:
        forward(x_1, x_2): Performs the forward pass of the BiCrossAttentionEncoder module.
    Returns:
        torch.Tensor: The average of the tensors obtained from the BiCrossAttention operation.
    """
    
    def __init__(self, d_in, d_out_kq, d_out_v, n_layers, reduction='mean'):
        super().__init__()
        self.layers = nn.ModuleList([BiCrossAttentionLayer(d_in, d_out_kq, d_out_v)
                                     for _ in range(n_layers)])
        self.reduction = reduction

    def forward(self, x_1, x_2):
        for layer in self.layers:
            x_1, x_2 = layer(x_1, x_2)

        if self.reduction == 'mean':
            return (x_1 + x_2) / 2
        elif self.reduction == 'concat':
            return torch.cat([x_1, x_2], dim=-1)


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


class CrossAttentionLayer(nn.Module):
    """
    CrossAttentionLayer module that applies multi-head attention mechanism between two tensors.
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

    def forward(self, x_0, x_1):
        x_0 = self.norm1(x_0)
        x_1 = self.norm1(x_1)
        x = x_0 + self.drop_path1(self.auto_att(x_0, x_1, x_1)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class CrossAttentionEncoder(nn.Module):
    """
    CrossAttentionEncoder module that applies multiple CrossAttentionLayer to the input tensors.
    Args:
        embed_dim (int): The dimension of the input tensor.
        num_heads (int, optional): The number of attention heads. Defaults to 1.
        n_layers (int, optional): The number of AutoAttentionLayer layers to apply. Defaults to 1.
    Returns:
        torch.Tensor: The output tensor after applying multiple AutoAttentionLayer layers.
    """
    
    
    def __init__(self, embed_dim, num_heads=1, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(embed_dim, num_heads=num_heads)
                                     for _ in range(n_layers)])

    def forward(self, x_0, x_1):
        for layer in self.layers:
            x = layer(x_0, x_1)
        return x
