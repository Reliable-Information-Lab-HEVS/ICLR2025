import torch

a = torch.rand(10,10).to('cuda')
print(a.device)