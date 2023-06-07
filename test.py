import torch
import numpy as np

foo = torch.cuda.is_available()
foo2 = torch.cuda.device_count() 
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(f'GPU available: {foo}.')
print(f'Number of GPUs: {foo2}.')

a = np.random.rand(1000, 1000)
b = torch.tensor(a, device=device)
c = b / 3
print(f'Newly computed tensor is on device: {c.device}.')
