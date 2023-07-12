import torch

a = torch.rand(10,10).to('cuda')
print(a.device)

property = torch.cuda.get_device_properties(0)
print(property)
print(property.total_memory)