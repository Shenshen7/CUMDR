import torch
import torch.nn as nn
import numpy as np

class Diffusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, temp=1):
        super(Diffusion, self).__init__()
        self.width = embed_dim
        self.dropout = dropout
        self.temp = temp
        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.proj = nn.Linear(self.width, self.width)

        self.sequence_pos_encoder = PositionalEncoding(self.width, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.width, self.sequence_pos_encoder)

        self.decoder = nn.Sequential(
            nn.Linear(self.width * 2, self.width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.width * 2, 1),
        )

    def forward(self, x, timesteps, text_feat, image_feat):
        

        cond_emb = self.embed_timestep(timesteps).squeeze(1)  # [batch_size, embed]
        i_weight = torch.einsum('bd,bnd->bn', [text_feat, image_feat]) # [batch, dim] * [batch, num_query, dim] -> [batch, num_query] 
        i_weight = torch.softmax(i_weight / self.temp, dim=-1)
        image_feat = torch.einsum('bnd,bn->bnd', [image_feat, i_weight]) # [batch, num_query, dim]
        q = self.q_proj(text_feat + cond_emb)  # [batch_size, dim]
        k = self.k_proj(image_feat + cond_emb.unsqueeze(1)) # [batch_size, num_query, dim]
        v = self.v_proj(image_feat + cond_emb.unsqueeze(1))  # [batch_size, num_query, dim]

        weight = torch.einsum('bd,bnd->bn', [q, k]) # [batch_size, num_query]
        weight = weight + x # [batch_size, num_query]
        weight = torch.softmax(weight, dim=-1) # [batch_size, num_query]
        new_emb = torch.einsum('bn,bnd->bd', [weight, v]) # [batch_size, dim]
        new_emb = self.proj(new_emb) # [batch_size, dim]

        # emb = torch.cat([new_emb, image_feat], dim=-1)  # [batch_size, dim*2]
        emb = torch.cat([new_emb.unsqueeze(1).repeat(1, image_feat.shape[1], 1), image_feat], dim=-1) # [batch_size, num_query, dim*2]
        # print(emb.shape)

        p = self.decoder(emb).squeeze(2)  # fc, then:[batch_size, num_query, 1]->[batch_size, num_query]
        p += weight # [batch_size, num_query]

        return p

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
