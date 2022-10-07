from torch.utils.cpp_extension import load

smo = load("smo", ["TLGCN/utils/smo_cuda.cpp", "TLGCN/utils/smo_cuda_kernel.cu"], verbose=True)