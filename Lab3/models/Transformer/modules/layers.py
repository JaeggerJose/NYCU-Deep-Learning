import torch.nn as nn
import torch
import math

#TODO1

def MyScaleDotProductAttention(q, k, v, attn_drop=0.1):
    ''' 
    q, k, v: (batch_size, num_image_tokens, dim)
    '''
    d_k = q.size(-1)
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    scale = attention_scores / math.sqrt(d_k) # average the dot product, prevent too large
    softmax = torch.softmax(scale, dim=-1) # to make the key distribution sum to 1 because it is a probability distribution
    attn = nn.Dropout(attn_drop)(softmax)
    output = torch.matmul(attn, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        # my implementation of Multi-Head Attention
        self.num_heads = num_heads
        self.dim = dim
        self.d_k = dim // num_heads
        self.d_v = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_tokens, _ = x.size()
        qkv = self.qkv_proj(x)  # shape: (B, N, 3*dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)
        
        attn = MyScaleDotProductAttention(q, k, v, self.attn_drop)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dim)
        output = self.proj(attn)
        return output


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    