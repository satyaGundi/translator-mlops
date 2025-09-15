
import torch

def is_gpu_avialable():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(1000, 1000, device="cuda")
        print("Tensor device:", x.device)