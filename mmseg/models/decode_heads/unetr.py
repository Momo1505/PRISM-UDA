import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..builder import build_loss


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features * 4)  # Expand dimension
        self.fc2 = nn.Linear(in_features * 4, in_features)  # Contract back
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, img_shape, dropout):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else [patch_size] * 3
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        
        # Calculate patch dimensions at initialization
        self.patch_dim = [
            img_shape[0] // self.patch_size[0],
            img_shape[1] // self.patch_size[1], 
            img_shape[2] // self.patch_size[2]
        ]
        self.n_patches = self.patch_dim[0] * self.patch_dim[1] * self.patch_dim[2]
        
        # Use separate patch sizes for each dimension 
        self.patch_embeddings = nn.Conv3d(
            in_channels=input_dim, 
            out_channels=embed_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Initialize position embeddings with fixed size
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.n_patches, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply patch embeddings
        x = self.patch_embeddings(x)  # Shape: [B, embed_dim, D', H', W']
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # Shape: [B, embed_dim, n_patches]
        
        # Transpose to get [B, n_patches, embed_dim]
        x = x.transpose(-1, -2)
        
        # Add position embeddings
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, img_shape, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, patch_size, img_shape, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, img_shape=(40, 256, 256), input_dim=1, output_dim=2, embed_dim=768, 
                 patch_size=(8, 16, 16), num_heads=12, dropout=0.1):  # Changed patch_size for anisotropic data
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else [patch_size] * 3
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        # Calculate patch dimensions
        self.patch_dim = [
            img_shape[0] // self.patch_size[0],  # 40//8 = 5
            img_shape[1] // self.patch_size[1],  # 256//16 = 16  
            img_shape[2] // self.patch_size[2]   # 256//16 = 16
        ]

        # Transformer Encoder
        self.transformer = Transformer(
            input_dim,
            embed_dim,
            patch_size,
            img_shape,
            num_heads,
            self.num_layers,
            dropout,
            self.ext_layers
        )

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv3DBlock(512, 256),
        )

        self.decoder9 = Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )
        
        # For binary segmentation, consider using BCEWithLogitsLoss
        loss_decode = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,

            )
            
        self.cross_entropy_loss = build_loss(loss_decode)

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        
        # Reshape transformer outputs back to 3D using fixed patch dimensions
        z3 = z3.transpose(-1, -2).view(z3.shape[0], self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(z6.shape[0], self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(z9.shape[0], self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(z12.shape[0], self.embed_dim, *self.patch_dim)

        # Process through decoders
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        
        # Handle dimension mismatch by interpolating z3 to match z0
        if z3.shape[2:] != z0.shape[2:]:
            z3 = F.interpolate(z3, size=z0.shape[2:], mode='trilinear', align_corners=False)
        
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
    
    def forward_train(self, inputs, gt_semantic_seg, seg_weight=None):
        return self.loss(inputs, gt_semantic_seg, seg_weight)
    
    def loss(self, inputs, gt_semantic_seg: torch.Tensor, seg_weight=None):
        """Calculate losses from a batch of inputs and data samples."""
        x = self.forward(inputs)
        
        losses = {}
        
        # Handle potential dimension issues
        if gt_semantic_seg.ndim == x.ndim:
            gt_semantic_seg = gt_semantic_seg.squeeze(1)
            
        loss_seg = self.cross_entropy_loss(
            x, 
            gt_semantic_seg.long(), 
            weight=seg_weight
        )
        
        losses['loss_seg'] = loss_seg
        losses['acc_seg'] = accuracy(x, gt_semantic_seg.long())
        
        return losses

    
def accuracy(pred:torch.Tensor, target:torch.Tensor, topk=1, thresh=None):
    """Calculate accuracy for multi-dimensional predictions.
    
    Args:
        pred (torch.Tensor): Predictions, shape (B, C, ...)
        target (torch.Tensor): Targets, shape (B, ...)
        topk (int | tuple[int]): Top-k accuracy levels.
        thresh (float, optional): Confidence threshold.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for _ in range(len(topk))]
        return accu[0] if return_single else accu

    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f"maxk {maxk} exceeds pred dimension {pred.size(1)}"

    # top-k along the class dimension
    pred_value, pred_label = pred.topk(maxk, dim=1)  
    # pred_label: (B, maxk, ...)
    # transpose -> (maxk, B, ...)
    pred_label = pred_label.transpose(0, 1)

    # Expand target for comparison
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))

    if thresh is not None:
        # pred_value is (B, maxk, ...)
        # permute to (maxk, B, ...) so it matches "correct"
        thresh_mask = (pred_value > thresh).permute(1, 0, *range(2, pred_value.ndim))
        correct = correct & thresh_mask

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    
    return res[0] if return_single else res
