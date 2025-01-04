import torch
import torch.nn as nn
import math

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=3, patch_size=(16,16,4), embed_dim=768, img_size=(224,224,32)):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.embed_dim = embed_dim

        # using Conv3d to do patch embeddings
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Initializing positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width, depth)
        x = self.proj(x)  # Shape: (batch_size, embed_dim, grid_h, grid_w, grid_d)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # shape: (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed  # add positional 
        return x  # return (batch_size, num_patches, embed_dim)

class VisionTransformer3D(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 patch_size=(16,16,4), 
                 embed_dim=768, 
                 img_size=(224,224,32), 
                 depth=12, 
                 num_heads=12, 
                 hidden_dim=3072, 
                 dropout=0.1):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim, img_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Output projection to match our input shape
        self.output_proj = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2])

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width, depth)
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)  
        x = self.norm(x) 
        # Proj back to patch size
        x = self.output_proj(x)  # Shape: (batch_size, num_patches, in_channels * patch_size[0] * patch_size[1] * patch_size[2])
        # reshape to original image dimensions
        batch_size = x.size(0)
        patch_size = self.patch_embed.patch_size
        grid_size = self.patch_embed.grid_size
        x = x.view(batch_size, self.patch_embed.num_patches, -1)  # (batch_size, num_patches, in_channels * patch_size)
        x = x.view(batch_size, 
                   grid_size[0], grid_size[1], grid_size[2], 
                   -1) 
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (batch_size, in_channels * patch_size, grid_h, grid_w, grid_d)
        # Reshaping patches back to original volume
        x = x.view(batch_size, 
                   -1, 
                   grid_size[0] * patch_size[0], 
                   grid_size[1] * patch_size[1], 
                   grid_size[2] * patch_size[2])  # (batch_size, in_channels, height, width, depth)
        return x  # Should match input shape now


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        x = x.transpose(0, 1)  # Shape: (num_patches, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        x = x + attn_output # add residual
        x = x.transpose(0, 1) 
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

