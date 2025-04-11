import torch



cuda_version = torch.version.cuda
print(cuda_version)

print(torch.cuda.is_available())

print(torch.backends.cudnn.version())



print('cuda' if torch.cuda.is_available() else 'cpu')