import torch
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")