import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available \U0001f44d")
    # Create two tensors and move them to GPU
    a = torch.randn(3, 3).cuda()
    b = torch.randn(3, 3).cuda()

    # Perform matrix multiplication (backed by CUDA kernels)
    c = torch.matmul(a, b)

    # Print result and device
    print("Result tensor:\n", c)
else:
    print("CUDA is not available.")
