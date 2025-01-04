import sys
import os
import torch
#testing file for 2d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vit_models.vit_2d import *

def test_vit_2d():


    # Instantiate the Vision Transformer model
    model = VisionTransformer2D(
        in_channels=3, 
        patch_size=16, 
        embed_dim=768, 
        img_size=224, 
        depth=12, 
        num_heads=12, 
        hidden_dim=3072, 
        dropout=0.1
    )

    # Forward pass
    pred = model(x)  
    assert pred.shape == x.shape, f"Output shape {pred.shape} does not match input shape {x.shape}"
    print("Forward pass successful. Output shape:", pred.shape)

    # Compute a dummy loss
    loss = torch.sum(pred)
    
    # Backward pass
    loss.backward()
    print("Backward pass successful. Loss computed:", loss.item())

if __name__ == "__main__":
    test_vit_2d()
