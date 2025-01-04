from vit_models import vit_2d, vit_3d, vit_4d

"""
Brief Approach/Example for 2d: 

PatchEmbedding: Converts an image into patch tokens
MultiHeadSelfAttention: Allows patches to attend to each other
FeedForward: Applies MLP to each patch embedding
TransformerEncoderLayer: Combines self-attention and feedforward layers
VisionTransformer2D	Complete Vision Transformer model for 2D images
"""