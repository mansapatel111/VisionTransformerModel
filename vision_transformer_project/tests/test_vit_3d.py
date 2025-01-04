import torch
import sys
import os
#esting file for 3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vit_models.vit_3d import VisionTransformer3D

def test_vit_3d():


    model = VisionTransformer3D(
        in_channels=3, 
        patch_size=(16, 16, 4), 
        embed_dim=256,  
        img_size=(224, 224, 32),  
        depth=6,  # Fewer transformer layers for memory
        num_heads=4, 
        hidden_dim=1024, 
        dropout=0.1
    )

    # Forward pass
    pred = model(x)  # Output should have the same shape as input
    assert pred.shape == x.shape, f"Output shape {pred.shape} does not match input shape {x.shape}"
    print("Forward pass successful. Output shape:", pred.shape)
    # Compute a dummy loss
    try:
        loss = torch.sum(pred)
        # Backward pass
        loss.backward()
        print("Backward pass successful. Loss computed:", loss.item())
    except Exception as e:
        print(f"Error during backward pass: {e}")

if __name__ == "__main__":
    test_vit_3d()
