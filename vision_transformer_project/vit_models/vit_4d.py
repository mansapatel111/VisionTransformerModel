import torch
import torch.nn as nn
import math

class PatchEmbedding4D(nn.Module):
    def __init__(self, in_channels=3, patch_size=(16,16,4,2), embed_dim=768, img_size=(224,224,32,4)):
        super(PatchEmbedding4D, self).__init__()
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3]
        )
        self.num_patches = 1
        for dim in self.grid_size:
            self.num_patches *= dim
        self.embed_dim = embed_dim

        # Assuming the temporal dimension is treated similar to depth
        # since conv4d is not directly supported in PyTorch, we will reshape the input to merge spatial and temporal dimensions and use conv3d
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                              kernel_size=(patch_size[0], patch_size[1], patch_size[2]*patch_size[3]),
                              stride=(patch_size[0], patch_size[1], patch_size[2]*patch_size[3]))
        # Initializing positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width, depth, time)
        batch_size, channels, height, width, depth, time = x.shape
        # Merge depth and time for Conv3d
        x = x.view(batch_size, channels, height, width, depth * time)
        x = self.proj(x)  # shape: (batch_size, embed_dim, grid_h, grid_w, grid_d)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed  # this will add positional encoding
        return x  # Sshape: (batch_size, num_patches, embed_dim)

class VisionTransformer4D(nn.Module):
    # change in size because of memory constraints
    def __init__(self, 
                 in_channels=3, 
                 patch_size=(16,16,4, 2), 
                 embed_dim=256, 
                 img_size=(224,224,32, 4), 
                 depth=6, 
                 num_heads=4, 
                 hidden_dim=1024, 
                 dropout=0.1):
        super(VisionTransformer4D, self).__init__()
        self.patch_embed = PatchEmbedding4D(in_channels, patch_size, embed_dim, img_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Output projection to match our input shape
        self.output_proj = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3])

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width, depth, time)
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)  
        x = self.norm(x) 
        # Projecting back to patch size
        x = self.output_proj(x)  # Shape: (batch_size, num_patches, in_channels * patch_size)
        # Reshape this to original image dimensions
        batch_size = x.size(0)
        patch_size = self.patch_embed.patch_size
        grid_size = self.patch_embed.grid_size
        x = x.view(batch_size, self.patch_embed.num_patches, -1)  # (batch_size, num_patches, in_channels * patch_size)
        x = x.view(batch_size, *grid_size, -1)  # (batch_size, grid_h, grid_w, grid_d, grid_t, in_channels * patch_size)
        # to (batch_size, in_channels * patch_size, grid_h, grid_w, grid_d, grid_t)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        # Reshape back to original dimensions
        x = x.view(batch_size, 
                   -1, 
                   grid_size[0] * patch_size[0], 
                   grid_size[1] * patch_size[1], 
                   grid_size[2] * patch_size[2],
                   grid_size[3] * patch_size[3])  
        return x  # Should now match input shape

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        x = x.transpose(0, 1)  # shape now: (num_patches, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = self.dropout(attn_output)
        x = x + attn_output 
        x = x.transpose(0, 1)  # (batch_size, num_patches, embed_dim)
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
        x = residual + x 
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

