import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import glob
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



# original code for vit model 2d
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
        # splits the image into patches and embeds them
        x = self.proj(x)  # shape: (batch_size, embed_dim, grid_size, grid_size)
        #flatterns the grid
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        #reorders dimensions
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        #adds positional inf to the patch embeddings.
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
        attn_output, _ = self.attention(x, x, x)  
        attn_output = self.dropout(attn_output)
        x = x + attn_output  
        x = x.transpose(0, 1)  #shape: (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        # exprojects the input to a higher-dimensional space.
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        # back to the original 
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        # potent reduce overfitting
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


# Autoregressive Vit Implementation   
class AutoregressiveViT(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 patch_size=16, 
                 embed_dim=768, 
                 img_size=224, 
                 depth=12, 
                 num_heads=12, 
                 hidden_dim=3072, 
                 dropout=0.1, 
                 forecast_steps=5):
        super(AutoregressiveViT, self).__init__()
        self.forecast_steps = forecast_steps
        self.patch_embed = PatchEmbedding2D(in_channels, patch_size, embed_dim, img_size)

        # Temporal Positional Encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1, forecast_steps, embed_dim))

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    def forward(self, x, steps=None):
        steps = steps or self.forecast_steps
        predictions = []

        for step in range(steps):
            # patches embedding
            x_patched = self.patch_embed(x)

            # adding temporal encoding
            x_patched += self.temporal_encoding[:, step % self.forecast_steps, :]

            # transformer layers with residual connections
            residual = x_patched
            for layer in self.layers:
                x_patched = layer(x_patched) + residual

            # Output we predict
            x_patched = self.norm(x_patched)
            out = self.output_proj(x_patched)

            # Reshape tje image
            batch_size = out.size(0)
            patch_size = self.patch_embed.patch_size
            grid_size = self.patch_embed.grid_size
            out = out.view(batch_size, grid_size, grid_size, -1)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = out.view(batch_size, -1, grid_size * patch_size, grid_size * patch_size)

            predictions.append(out)

            # Trying Scheduled Sampling
            if self.training and torch.rand(1).item() < 0.5:  # 50% chance to use model's output
                x = out
            else:
                x = x  # dummy mde uses x, replacingss with dataset ground truth 

        return torch.stack(predictions, dim=1)  # predicted shape: (batch_size, steps, channels, H, W)

#  Evaluation Metrics
# def evaluate(model, dataloader, criterion, device, forecast_steps=5):

def compute_metrics(predictions, targets):
    mse = F.mse_loss(predictions, targets).item()
    rmse = mse ** 0.5
    mae = F.l1_loss(predictions, targets).item()

    # future steps maybe Correlation Coefficient implementation

    return {"MSE": mse, "RMSE": rmse, "MAE": mae}



# Training Loop with Multi Scale Loss 

def multi_scale_loss(predictions, targets, criterion):
    pixel_loss = criterion(predictions, targets)
    downsampled_preds = nn.functional.avg_pool2d(predictions, kernel_size=4)
    downsampled_targets = nn.functional.avg_pool2d(targets, kernel_size=4)
    structure_loss = criterion(downsampled_preds, downsampled_targets)
    return 0.7 * pixel_loss + 0.3 * structure_loss


def train(model, dataloader, criterion, optimizer, device, forecast_steps=5):
    model.train()
    total_loss = 0
    all_metrics = {"MSE": 0, "RMSE": 0, "MAE": 0}

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs, steps=forecast_steps)

        # Multi Scale Loss applied hereeeeeeee
        loss = sum(multi_scale_loss(outputs[:, step], targets[:, step], criterion) 
                   for step in range(forecast_steps)) / forecast_steps

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # calculating metrics
        metrics = compute_metrics(outputs, targets)
        for key in all_metrics:
            all_metrics[key] += metrics[key]

    num_batches = len(dataloader)
    avg_metrics = {key: value / num_batches for key, value in all_metrics.items()}
    avg_metrics["Loss"] = total_loss / num_batches

    return avg_metrics


# Dataset Loading 


def load_dataset(data_dir="2d/"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pth")))  # ensuring sorted order
    dataset = [torch.load(file) for file in files]  
    dataset = torch.stack(dataset) 
    print(f"Loaded dataset with shape: {dataset.shape}")
    return dataset

def split_dataset(dataset, test_size=0.2):
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)
    return train_data, test_data

def create_dataloaders(train_data, test_data, batch_size=8):
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoregressiveViT(in_channels=1, img_size=224).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Dummy dataset for git push    
    dummy_data = torch.rand(8, 1, 224, 224).to(device)
    dummy_targets = torch.rand(8, 5, 1, 224, 224).to(device)  # Predicting 5 future steps
    dataloader = [(dummy_data, dummy_targets)]

    # Actual dataset loading
    # dataset = load_dataset("2d/")
    # train_data, test_data = split_dataset(dataset)
    # train_loader, test_loader = create_dataloaders(train_data, test_data)

    # Training
    for epoch in range(10):
        metrics = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {metrics['Loss']:.4f}, RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")

if __name__ == "__main__":
    main()
