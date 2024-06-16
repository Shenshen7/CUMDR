import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DisAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, mu_heads_num+logsig_heads_num, n, dim_heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C/2))

        mu = x[:,:,0,:]
        logsigma = x[:,:,1,:]
        mu = self.mu_proj(mu)
        mu = self.mu_proj_drop(mu)
        logsigma = self.logsig_proj(logsigma)
        logsigma = self.logsig_proj_drop(logsigma)
        return mu, logsigma, attn


class DisTrans(nn.Module):
    def __init__(
        self,
        dim,
        indim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.logsig_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        # x = self.fc(x)
        # x_ = self.act(x)
        # x_ = self.norm1(x_)
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_, mask=mask)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        return mu, logsigma, attn
    
def gaussian_modeling(
    image_embeds,
    extend_image_masks,
    text_embeds,
    extend_text_masks,
    img_gau_encoder,
    txt_gau_encoder,
    mu_num,
    sample_num

):
    if img_gau_encoder !=None:
        img_mu, img_logsigma, _ = img_gau_encoder(image_embeds, mask=extend_image_masks)
        z = [img_mu] * mu_num
        for i in range(sample_num):
            eps = torch.randn(img_mu.shape[0], img_mu.shape[1], img_mu.shape[2], device=img_mu.device)
            z1 = img_mu + torch.exp(img_logsigma) * eps
            z.append(z1)
        image_embeds = torch.cat(z)
    else:
        img_mu = img_logsigma = image_embeds =None
    if txt_gau_encoder != None:
        txt_mu, txt_logsigma, _ = txt_gau_encoder(text_embeds, mask=extend_text_masks)
        z = [txt_mu] * mu_num
        for i in range(sample_num):
            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            z1 = txt_mu + torch.exp(txt_logsigma) * eps
            z.append(z1)
        text_embeds = torch.cat(z)
    else:
        txt_mu = txt_logsigma = image_embeds = None

    return image_embeds, text_embeds, img_mu, img_logsigma, txt_mu, txt_logsigma