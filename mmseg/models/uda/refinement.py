import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple,trunc_normal_
import math
from einops import rearrange
import pdb
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
                 img_size=1024,
                 patch_size=16,
                 stride=16,
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
            stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        token = math.floor((img_size[0] + 2 * (patch_size[0] // 2) - patch_size[0]) / stride)
        self.num_tokens= token 

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x


class ConvWithBn(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: int=3, stride: int = 1, padding:int=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layers(x)


class DeconvWithBN(nn.Module):
    def __init__(self,in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvWithBn(in_channels=in_channels,out_channels=out_channels)
        )
    
    def forward(self,x):
        return self.layers(x)

class Deconv(nn.Module):
    def __init__(self,in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        )
    
    def forward(self,x):
        return self.deconv(x)

class Classification(nn.Module):
    def __init__(self,in_channels: int,out_channels: int, kernel_size: int=1, stride: int = 1, padding:int=0):
        super().__init__()
        self.out = nn.Sequential(
            ConvWithBn(in_channels=in_channels,out_channels=in_channels//2),
            ConvWithBn(in_channels=in_channels//2,out_channels=out_channels),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        )

    def forward(self,x):
        return self.out(x)

class Upsample(nn.Module):
    def __init__(self,in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(ConvWithBn(in_channels=in_channels,out_channels=in_channels//2),
                                ConvWithBn(in_channels=in_channels//2,out_channels=in_channels),
                                Deconv(in_channels=in_channels,out_channels=out_channels))

    def forward(self,x):
        return self.up(x)

class DeconvAndUp(nn.Module):
    def __init__(self,m:nn.ModuleList):
        super().__init__()
        self.m = m
    def forward(self,skip,latent):
        skip = self.m[0](skip)
        x = torch.cat([skip,latent],dim=1)
        return self.m[1](x)

class Decoder(nn.Module):
    def __init__(self,embed_dim,num_token):
        super().__init__()
        self.z_12 = nn.Sequential(DeconvWithBN(embed_dim,out_channels=256),DeconvWithBN(256,out_channels=256))

        self.z_9 = DeconvAndUp(nn.ModuleList([
            nn.Sequential(DeconvWithBN(in_channels=embed_dim,out_channels=embed_dim//2),DeconvWithBN(in_channels=embed_dim//2,out_channels=256)),
            Upsample(in_channels=512,out_channels=128)
        ]))

        self.z_6 = DeconvAndUp(nn.ModuleList([
            nn.Sequential(DeconvWithBN(in_channels=embed_dim,out_channels=embed_dim//2),DeconvWithBN(in_channels=embed_dim//2,out_channels=256),DeconvWithBN(in_channels=256,out_channels=128)),
            Upsample(in_channels=256,out_channels=64)
        ]))
        self.z_3 = DeconvAndUp(nn.ModuleList([
            nn.Sequential(DeconvWithBN(in_channels=embed_dim,out_channels=embed_dim//2),DeconvWithBN(in_channels=embed_dim//2,out_channels=256),DeconvWithBN(in_channels=256,out_channels=128),DeconvWithBN(in_channels=128,out_channels=64)),
            Upsample(in_channels=128,out_channels=32)
        ])
)
        self.input_skip = nn.Sequential(
            ConvWithBn(in_channels=1,out_channels=16),
            ConvWithBn(in_channels=16,out_channels=32),
        )

        self.classification = Classification(in_channels=64,out_channels=2)
        self.num_token = num_token

    def forward(self,hidden:list):
        latent = rearrange(hidden[-1], "b (h w) c -> b c h w", h=self.num_token)
        z_12 = self.z_12(latent) # b,256,128,128
        latent = rearrange(hidden[-2], "b (h w) c -> b c h w", h=self.num_token)
        z_9 = self.z_9(latent,z_12)
        latent = rearrange(hidden[-3], "b (h w) c -> b c h w", h=self.num_token)
        z_6 = self.z_6(latent,z_9)

        latent = rearrange(hidden[-4], "b (h w) c -> b c h w", h=self.num_token)
        z_3 = self.z_3(latent,z_6)

        x = torch.cat([
            self.input_skip(hidden[-5]),
            z_3
        ],
            dim=1)
        return self.classification(x)



class Encoder(nn.Module):
    def __init__(self,device="cpu",num_blocks=12,img_size=1024,
                 patch_size=32,
                 stride=32,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        self.sam_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        self.pl_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        num_seq = self.pl_embed.num_tokens ** 2
        self.pos_embed = nn.Embedding(num_seq,embed_dim)
        self.register_buffer("positions",torch.arange(num_seq,device=device))

        self.encode = nn.ModuleList([Guidance(embed_dim) for _ in range(num_blocks)])
        self.decode = Decoder(embed_dim=embed_dim,num_token=self.pl_embed.num_tokens)


    def forward(self, sam, pl_source):
        positions = self.pos_embed(self.positions)
        sam_embed = self.sam_embed(sam) + positions
        pl_embed = self.pl_embed(pl_source) + positions

        hidden = [sam+pl_source]

        for i,layer in enumerate(self.encode):
            sam,pl_source = layer(sam_embed,pl_embed)
            if (i+1) % 3 == 0 :
                hidden.append(sam + pl_source)
        
        return self.decode(hidden)


