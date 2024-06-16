import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.WQ = nn.Linear(input_dim, input_dim)
        self.WK = nn.Linear(input_dim, input_dim)
        self.WV = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
        
    def forward(self, q, k, v):
        # Linearly project queries, keys, and values
        queries = self.WQ(q)
        keys = self.WK(k)
        values = self.WV(v)
        
        # Split into multiple heads
        queries = torch.chunk(queries, self.num_heads, dim=-1)
        keys = torch.chunk(keys, self.num_heads, dim=-1)
        values = torch.chunk(values, self.num_heads, dim=-1)
        
        # Concatenate heads
        queries = torch.cat(queries, dim=0)
        keys = torch.cat(keys, dim=0)
        values = torch.cat(values, dim=0)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, values)
        
        # Split and concat heads for the output
        output = torch.chunk(output, self.num_heads, dim=0)
        output = torch.cat(output, dim=-1)
        
        # Linearly project the output and apply a final fully connected layer
        output = self.fc(output)
        
        return output

class feedForward(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out
    
class addNorm(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x, y):
        return self.norm(x + y)

class coAttenTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_head) -> None:
        super().__init__()
        self.fw = feedForward(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_head)
        # self.attn = nn.MultiheadAttention(embed_dim, num_head)
        self.norm1 = addNorm(embed_dim)
        self.norm2 = addNorm(embed_dim)
    def forward(self, q, v):
        # out = self.attn(q, v, v, need_weights=False, attn_mask=None)[0]
        out = self.attn(q, v, v)
        out = self.norm1(q, out)
        out = self.norm2(out, self.fw(out))
        return out


class coAttenTransformer(nn.Module):
    def __init__(self, embed_dim, num_head, num_layer) -> None:
        super().__init__()
        self.blocks1 = nn.Sequential(*[coAttenTransformerBlock( embed_dim, num_head) for _ in range(num_layer)])
        self.blocks2 = nn.Sequential(*[coAttenTransformerBlock( embed_dim, num_head) for _ in range(num_layer)])
    def forward(self, mod1, mod2):
        for block1, block2 in zip(self.blocks1, self.blocks2):
            temp = block1(mod1, mod2)
            mod2 = block2(mod2, mod1)
            mod1 = temp
        return mod1, mod2



