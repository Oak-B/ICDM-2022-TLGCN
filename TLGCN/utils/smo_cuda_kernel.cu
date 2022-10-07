#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#define BLOCK_ROWS 32
#define BLOCK_COLS 32
#define BLOCK_SIZE 1024

/*
get the number of each col's values
*/
__global__ void get_col_num_cuda_kernel(const int* __restrict__ Aj, int* __restrict__ Bp, const int nnz){
    int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_col < nnz){
        int col_id = Aj[thread_col] + 1;
        atomicAdd(&Bp[col_id], 1);
    }
}

__global__ void get_indices_data_cuda_kernel(const int* __restrict__ Ap, const int* __restrict__ Aj, const float* __restrict__ Ax,
    int* __restrict__ Bp, int* __restrict__ Bi, float* __restrict__ Bx, const int n_row){
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + thread_row;
    int col = blockIdx.x * blockDim.x + thread_col;
    int start_index = Ap[row];
    int num_entity = Ap[row + 1] - start_index;
    if(col < num_entity){
        int o_col = Aj[start_index + col];
        int dest = atomicAdd(&Bp[o_col], 1);
        Bi[dest] = row;
        Bx[dest] = Ax[start_index + col];
    }
}

/*
Reference https://github.com/pytorch/pytorch/blob/5bdc4db26e91db9639e85b1c8564b0a99c4f1075/caffe2/utils/math/transpose.cu
*/
__global__ void BatchTranspose2DCUDAKernel(const int H, const int W, const int dh, const int dw,
    const float* __restrict__ X, float* __restrict__ Y) {
    __shared__ float tile[BLOCK_COLS][BLOCK_COLS + 1];
    int n = blockIdx.x / (dh * dw);
    int k = blockIdx.x % (dh * dw);
    int r = k / dw;
    int c = k % dw;
    int offset = n * H * W;
    int x = c * BLOCK_COLS + threadIdx.x;
    int y = r * BLOCK_COLS + threadIdx.y;
    if (x < W) {
        for (int i = 0; threadIdx.y + i < BLOCK_COLS && y + i < H; i += BLOCK_ROWS) {
            tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
        }
    }
    __syncthreads();
    x = r * BLOCK_COLS + threadIdx.x;
    y = c * BLOCK_COLS + threadIdx.y;
    if (x < H) {
        for (int i = 0; threadIdx.y + i < BLOCK_COLS && y + i < W; i += BLOCK_ROWS) {
            Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void smo_s_d_d_mm_cuda_kernel(const int* __restrict__ Ap, const int* __restrict__ Aj, const float* __restrict__ Ax,
    const float* __restrict__ dense, float* __restrict__ result, const int s_n_row, const int d_n_col){
    __shared__ int Aj_tile[BLOCK_ROWS][BLOCK_COLS];
    __shared__ float Ax_tile[BLOCK_ROWS][BLOCK_COLS];
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + thread_row;
    int col = blockIdx.x * blockDim.x + thread_col;
    if(row < s_n_row){
        int start_index = Ap[row];
        int num_entity = Ap[row + 1] - start_index;
        float value = 0.0;
        for(int m = 0; m < (num_entity - 1) / BLOCK_COLS + 1; m++){
            if(m * BLOCK_COLS + thread_col < num_entity){
                Aj_tile[thread_row][thread_col] = Aj[start_index + m * BLOCK_COLS + thread_col];
                Ax_tile[thread_row][thread_col] = Ax[start_index + m * BLOCK_COLS + thread_col];
            }
            // __syncthreads();
            for(int i = 0; i < BLOCK_COLS && m * BLOCK_COLS + i < num_entity; i++){
                value += Ax_tile[thread_row][i] * dense[Aj_tile[thread_row][i] * d_n_col + col];
            }
        }
        if(col < d_n_col){
            result[row * d_n_col + col] = value;
        }
    }

}

__global__ void smo_d_d_s_mm_cuda_kernel(const float* __restrict__ dense1, const float* __restrict__ dense2, const int* __restrict__ Ap, 
    const int* __restrict__ Aj, float* __restrict__ result, const int d1_n_row, const int d1_n_col, const int d2_n_col){
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + thread_row;
    int col = blockIdx.x * blockDim.x + thread_col;
    if(row < d1_n_row){
        int start_index = Ap[row];
        int num_entity = Ap[row + 1] - start_index;
        int block_col = blockIdx.x * blockDim.x;
        if(col < num_entity){
            float value = 0.0;
            for(int m = 0; m < d1_n_col; m++){
                value += dense1[row * d1_n_col + m] * dense2[m * d2_n_col + Aj[start_index + col]];
            }
            result[start_index + col] = value;
        }
    }    
}


__global__ void smo_d_s_d_mm_cuda_kernel(const float* __restrict__ dense_t, const int* __restrict__ Ap_t, const int* __restrict__ Aj_t,
    const float* __restrict__ Ax_t, float* __restrict__ result, const int d_t_n_col, const int s_t_n_row){
    __shared__ int Aj_t_tile[BLOCK_ROWS][BLOCK_COLS];
    __shared__ float Ax_t_tile[BLOCK_ROWS][BLOCK_COLS];
    // __shared__ float result_tile[BLOCK_ROWS][BLOCK_COLS + 1];
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + thread_row;
    int col = blockIdx.x * blockDim.x + thread_col;
    if(row < s_t_n_row){
        int start_index = Ap_t[row];
        int num_entity = Ap_t[row + 1] -start_index;
        float value = 0.0;
        for(int m = 0; m < (num_entity - 1) / BLOCK_COLS + 1; m++){
            if(m * BLOCK_COLS + thread_col < num_entity){
                Aj_t_tile[thread_row][thread_col] = Aj_t[start_index + m * BLOCK_COLS + thread_col];
                Ax_t_tile[thread_row][thread_col] = Ax_t[start_index + m * BLOCK_COLS + thread_col];
            }
            // __syncthreads();
            for(int i = 0; i < BLOCK_COLS && m * BLOCK_COLS + i < num_entity; i++){
                value += Ax_t_tile[thread_row][i] * dense_t[Aj_t_tile[thread_row][i] * d_t_n_col + col];
            }
        }
        if(col < d_t_n_col){
            result[col * s_t_n_row + row] = value;
        }
    }
}

__global__ void smo_s_d_s_add_cuda_kernel(const int* __restrict__ Aj, const float* __restrict__ Ax,
    const float* __restrict__ dense_vector, float* __restrict__ result, const int nnz){
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if(col < nnz){
        int col_id = Aj[col];
        float value = Ax[col];
        float value_d = dense_vector[col_id];
        result[col] = value + value_d;
    }
}

__global__ void smo_s_sum_cuda_kernel(const int* __restrict__ Aj, const float* __restrict__ Ax,
    float* __restrict__ result, const int nnz){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < nnz){
        int col_id = Aj[col];
        float value = Ax[col];
        atomicAdd(&result[col_id], value);
    }
}

std::vector<torch::Tensor> smo_s_transpose_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col){
    int nnz = Ax.size(0);
    auto Bp = torch::zeros({n_col + 1}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    auto Bi = torch::zeros({nnz}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    auto Bx = torch::zeros({nnz}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    dim3 dimBlock1(BLOCK_SIZE);
    dim3 dimGrid1((nnz - 1) / BLOCK_SIZE + 1);
    get_col_num_cuda_kernel<<<dimGrid1, dimBlock1>>>(Aj.data<int>(), Bp.data<int>(), nnz);

    thrust::inclusive_scan(thrust::device, Bp.data<int>() + 1, Bp.data<int>() + 1 + n_col, Bp.data<int>() + 1);
    
    auto temp_Bp = torch::zeros({n_col + 1}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    thrust::copy(thrust::device, Bp.data<int>(), Bp.data<int>() + n_col + 1, temp_Bp.data<int>());

    dim3 dimBlock2(BLOCK_SIZE);
    dim3 dimGrid2((n_col * 0.5 - 1) / BLOCK_SIZE + 1, n_row);
    get_indices_data_cuda_kernel<<<dimGrid2, dimBlock2>>>(Ap.data<int>(), Aj.data<int>(), Ax.data<float>(),
        temp_Bp.data<int>(), Bi.data<int>(), Bx.data<float>(), n_row);
    return {Bp, Bi, Bx};
}

/*
matrix transpose
*/
torch::Tensor smo_d_transpose_cuda(torch::Tensor X, int H, int W){
    auto Y = torch::zeros({W, H}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    const int dh = (H - 1) / BLOCK_COLS + 1;
    const int dw = (W - 1) / BLOCK_COLS + 1;
    BatchTranspose2DCUDAKernel<<<dim3(dh * dw), dim3(BLOCK_COLS, BLOCK_ROWS)>>>(H, W, dh, dw, X.data<float>(), Y.data<float>());
    return Y;
}


torch::Tensor smo_s_d_d_mm_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense, int s_n_row, int d_n_col){
    auto result = torch::zeros({s_n_row, d_n_col}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    dim3 dimBlock(BLOCK_COLS, BLOCK_ROWS);
    dim3 dimGrid((d_n_col - 1) / BLOCK_COLS + 1, (s_n_row - 1) / BLOCK_ROWS + 1);
    smo_s_d_d_mm_cuda_kernel<<<dimGrid, dimBlock>>>(Ap.data<int>(), Aj.data<int>(), Ax.data<float>(),
        dense.data<float>(), result.data<float>(), s_n_row, d_n_col);
    return result;
}

torch::Tensor smo_d_d_s_mm_cuda(torch::Tensor dense1, torch::Tensor dense2, torch::Tensor Ap, torch::Tensor Aj, int d1_n_row, int d1_n_col, int d2_n_col){
    int nnz = Aj.size(0);
    auto result = torch::zeros({nnz}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((d2_n_col * 0.5 - 1) / BLOCK_SIZE + 1, d1_n_row);
    smo_d_d_s_mm_cuda_kernel<<<dimGrid, dimBlock>>>(dense1.data<float>(), dense2.data<float>(), 
        Ap.data<int>(), Aj.data<int>(), result.data<float>(), d1_n_row, d1_n_col, d2_n_col);
    return result;
}


torch::Tensor smo_d_s_d_mm_cuda(torch::Tensor dense_t, torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int d_t_n_row, int d_t_n_col, int s_n_col){
    auto result = torch::zeros({d_t_n_col, s_n_col}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    std::vector<torch::Tensor> s_t = smo_s_transpose_cuda(Ap, Aj, Ax, d_t_n_row, s_n_col);
    dim3 dimBlock(BLOCK_COLS, BLOCK_ROWS);
    dim3 dimGrid((d_t_n_col - 1) / BLOCK_COLS + 1, (s_n_col - 1) / BLOCK_ROWS + 1);
    smo_d_s_d_mm_cuda_kernel<<<dimGrid, dimBlock>>>(dense_t.data<float>(), s_t[0].data<int>(), s_t[1].data<int>(),
        s_t[2].data<float>(), result.data<float>(), d_t_n_col, s_n_col);
    return result;
}

torch::Tensor smo_s_d_s_add_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense_vector, int s_n_row, int s_n_col){
    int nnz = Aj.size(0);
    auto result = torch::zeros({nnz}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((nnz - 1) / BLOCK_SIZE + 1);
    smo_s_d_s_add_cuda_kernel<<<dimGrid, dimBlock>>>(Aj.data<int>(), Ax.data<float>(), dense_vector.data<float>(), result.data<float>(), nnz);
    return result;
}

torch::Tensor smo_s_sum_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col){
    int nnz = Aj.size(0);
    auto result = torch::zeros({n_col}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((nnz - 1) / BLOCK_SIZE + 1);
    smo_s_sum_cuda_kernel<<<dimGrid, dimBlock>>>(Aj.data<int>(), Ax.data<float>(), result.data<float>(), nnz);
    return result;
}