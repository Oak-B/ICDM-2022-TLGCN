from torch.utils.cpp_extension import load

smo = load("smo", ["FAE/utils/smo_cuda.cpp", "FAE/utils/smo_cuda_kernel.cu"], verbose=True)