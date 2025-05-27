import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple,trunc_normal_
import math
from einops import rearrange

class Guidance(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_sam = nn.Linear(dim, dim, bias=qkv_bias) 
        self.q_pl_source = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv_sam = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_pl_source = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop_sam = nn.Dropout(attn_drop) 
        self.attn_pl_source = nn.Dropout(attn_drop)

        self.proj_sam = nn.Linear(dim, dim) 
        self.proj_pl_source = nn.Linear(dim, dim)

        self.proj_drop_sam = nn.Dropout(proj_drop)
        self.proj_pl_source = nn.Dropout(proj_drop) 


    def forward(self, sam,pl_source):
        B, N, C = sam.shape
        q_sam = self.q_sam(sam).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3).contiguous()

        q_pl_source = self.q_pl_source(pl_source).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3).contiguous()
        

        kv_sam = self.kv_sam(sam).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_sam, v_sam = kv_sam[0], kv_sam[1]

        kv_pl_source = self.kv_pl_source(pl_source).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_pl_source , v_pl_source  = kv_pl_source[0], kv_pl_source[1]

        attn_sam = (q_pl_source @ k_sam.transpose(-2, -1).contiguous()) * self.scale
        attn_sam = attn_sam.softmax(dim=-1)
        attn_sam = self.attn_drop_sam(attn_sam)

        sam_attention = (attn_sam @ v_sam).transpose(1, 2).contiguous().reshape(B, N, C)
        sam_attention = self.proj_sam(sam_attention)
        guidance_sam = self.proj_drop_sam(sam_attention + sam)

        attn_pl_source = (q_sam @ k_pl_source.transpose(-2, -1).contiguous()) * self.scale
        attn_pl_source = attn_pl_source.softmax(dim=-1)
        attn_pl_source = self.attn_pl_source(attn_pl_source)

        attn_pl_source = (attn_pl_source @ v_pl_source).transpose(1, 2).contiguous().reshape(B, N, C)
        attn_pl_source = self.proj_sam(attn_pl_source)
        guidance_pl_source = self.proj_drop_sam(attn_pl_source + pl_source)

        return guidance_sam,guidance_pl_source

def init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=256,
                 patch_size=7,
                 stride=4,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        token = math.floor((img_size[0] + 2 * (patch_size[0] // 2) - patch_size[0]) / stride) + 1
        self.num_tokens= token ** 2

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self,device,num_blocks=6,img_size=256,
                 patch_size=7,
                 stride=4,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        self.sam_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        self.pl_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        num_seq = self.pl_embed.num_tokens
        self.pos_embed = nn.Embedding(num_seq,embed_dim)
        self.register_buffer("positions",torch.arange(num_seq,device=device))

        self.encode = nn.ModuleList([Guidance(embed_dim) for _ in range(num_blocks)])


    def forward(self, sam, pl_source):
        positions = self.pos_embed(self.positions)
        sam_embed = self.sam_embed(sam) + positions
        pl_embed = self.pl_embed(pl_source) + positions

        for layer in self.encode:
            sam,pl_source = layer(sam_embed,pl_embed)
        return sam , pl_source

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder,img_size,stride):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.num_token = math.floor((img_size + 2 * (patch_size // 2) - patch_size) / stride) + 1

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.img_size = img_size

    def forward(self, x):
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.num_token)

        return nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

class EncodeDecode(nn.Module):
    def __init__(self,device,img_size=256,
                 patch_size=7,
                 stride=4,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        self.encode = Encoder(device,img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=2*embed_dim,out_channels=embed_dim,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim//2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim//2,out_channels=embed_dim,kernel_size=1)
        )
        self.decode = DecoderLinear(n_cls=19,patch_size=patch_size,d_encoder=embed_dim,img_size=img_size,stride=stride)

        self.num_token = self.decode.num_token

        self.apply(init_weights)
        
    def forward(self, sam, pl_source):

        sam = F.interpolate(sam.float(),size=(256,256),mode='bilinear', align_corners=False)
        pl_source = F.interpolate(pl_source.float(),size=(256,256),mode='bilinear', align_corners=False)
        sam_latent,pl_source_latent = self.encode(sam,pl_source)
        sam_latent = rearrange(sam_latent, "b (h w) c -> b c h w", h=self.num_token)
        pl_source_latent = rearrange(pl_source_latent, "b (h w) c -> b c h w", h=self.num_token)
        x = torch.cat([sam_latent,pl_source_latent],dim=1) 
        x = self.fuse(x)
        x = rearrange(x, "b c h w -> b (h w) c", h=self.num_token)
        return self.decode(x) * (sam + pl_source)
