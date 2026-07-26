import torch
import torch.nn as nn

class HighPassFilter(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[[0., -1., 0.],
                                 [-1., 5., -1.],
                                 [0., -1., 0.]]]])
        self.kernel = kernel.repeat(3, 1, 1, 1)
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel, padding=1, groups=3)
