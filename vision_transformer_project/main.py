import torch
from vit_models.vit_2d import VisionTransformer2D
from vit_models.vit_3d import VisionTransformer3D
from vit_models.vit_4d import VisionTransformer4D

def main_1():

    
    x_1 = torch.rand(4, 3, 224, 224).to(device)
    
    # 2D Data Test
    print("Testing 2D Vision Transformer...")
    model_2d = VisionTransformer2D().to(device)
    pred_2d = model_2d(x_1)
    print("2D Output Shape:", pred_2d.shape)
    # Compute a dummy loss
    loss = torch.sum(pred_2d)
    
    # Backward pass
    loss.backward()
    print("Backward pass successful. Loss computed:", loss.item())


def main_2():
    # 3D Data Test
    x_2 = torch.rand(4, 3, 224, 224, 32).to(device)
    print("Testing 3D Vision Transformer...")
    model_3d = VisionTransformer3D().to(device)
    pred_3d = model_3d(x_2)
    print("3D Output Shape:", pred_3d.shape)
     # Compute a dummy loss
    loss = torch.sum(pred_3d)
    
    # Backward pass
    loss.backward()
    print("Backward pass successful. Loss computed:", loss.item())

def main_3():

   # 4D Data Test
    x_3 = torch.rand(4, 3, 224, 224, 32, 4).to(device)
    print("Testing 4D Vision Transformer...")
   
    model_4d = VisionTransformer4D().to(device)
    pred_4d = model_4d(x_3)
    print("4D Output Shape:", pred_4d.shape)
     # Compute a dummy loss
    loss = torch.sum(pred_4d)
    
    # Backward pass
    loss.backward()
    print("Backward pass successful. Loss computed:", loss.item())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # creating separate functions due to memory issue
    main_1()
    main_2()
    main_3()
