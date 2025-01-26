import torch
import torch.nn as nn

class ImageProjector(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super(ImageProjector, self).__init__()
        self.projection = nn.Linear(latent_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        # x shape: [bs, 1280, 8, 8]
        pooled_x = torch.mean(x, dim=[2, 3]) # Shape: [batch_size, 1280]
        # pooled_x = x.flatten(1)
        res = self.projection(pooled_x)
        return res