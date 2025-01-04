import torch
import torch.nn as nn
import math

class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, img_size=224):
        super(PatchEmbedding2D, self).__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim

        # Using Conv2d to perform patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Initializing positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        x = self.proj(x)  # shape: (batch_size, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed  # adding pos encoding
        return x  # (batch_size, num_patches, embed_dim)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        x = x.transpose(0, 1)  # Shape: (num_patches, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = self.dropout(attn_output)
        x = x + attn_output  
        x = x.transpose(0, 1)  #shape: (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x  # residual connect
        x = self.norm(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        x = self.mhsa(x)
        x = self.ffn(x)
        return x

class VisionTransformer2D(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 patch_size=16, 
                 embed_dim=768, 
                 img_size=224, 
                 depth=12, 
                 num_heads=12, 
                 hidden_dim=3072, 
                 dropout=0.1):
        super(VisionTransformer2D, self).__init__()
        self.patch_embed = PatchEmbedding2D(in_channels, patch_size, embed_dim, img_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Output projection to match our input shape
        self.output_proj = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x) 
        x = self.norm(x)
        #back to patch size
        x = self.output_proj(x)  # shape: (batch_size, num_patches, in_channels * patch_size * patch_size)
        # reshape to original image dimensions
        batch_size = x.size(0)
        patch_size = self.patch_embed.patch_size
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size
        x = x.view(batch_size, num_patches, -1)  # (batch_size, num_patches, in_channels * patch_size * patch_size)
        x = x.view(batch_size, grid_size, grid_size, -1)  # (batch_size, grid_size, grid_size, in_channels * patch_size * patch_size)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch_size, in_channels * patch_size * patch_size, grid_size, grid_size)
        # reshaping patches back to image
        x = x.view(batch_size, 
                   -1, 
                   grid_size * patch_size, 
                   grid_size * patch_size)  # (batch_size, in_channels, height, width)
        return x  #  match input shape

