#include <torch/extension.h>
#include <vector>
#include <torch/torch.h>

std::vector<torch::Tensor> smo_s_transpose_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col);
torch::Tensor smo_d_transpose_cuda(torch::Tensor X, int H, int W);
torch::Tensor smo_s_d_d_mm_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense, int s_n_row, int d_n_col);
torch::Tensor smo_d_d_s_mm_cuda(torch::Tensor dense1, torch::Tensor dense2, torch::Tensor Ap, torch::Tensor Aj, int d1_n_row, int d1_n_col, int d2_n_col);
torch::Tensor smo_d_s_d_mm_cuda(torch::Tensor dense_t, torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int d_t_n_row, int d_t_n_col, int s_n_col);
torch::Tensor smo_s_d_s_add_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense_vector, int s_n_row, int s_n_col);
torch::Tensor smo_s_sum_cuda(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
sparse matrix transpose
Ap: row pointer
Aj: column indices
Ax: nonzeros
n_row: number of rows
n_col: number of columns
return: the transpose (Bp, Bi, Bx)
*/
std::vector<torch::Tensor> smo_s_transpose(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col){
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    CHECK_INPUT(Ax);
    return smo_s_transpose_cuda(Ap, Aj, Ax, n_row, n_col);
}

torch::Tensor smo_d_transpose(torch::Tensor X, int H, int W){
    CHECK_INPUT(X);
    return smo_d_transpose_cuda(X, H, W);
}

/*
spare * dense = dense
*/
torch::Tensor smo_s_d_d_mm(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense, int s_n_row, int d_n_col){
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    CHECK_INPUT(Ax);
    CHECK_INPUT(dense);
    return smo_s_d_d_mm_cuda(Ap, Aj, Ax, dense, s_n_row, d_n_col);
}

torch::Tensor smo_d_d_s_mm(torch::Tensor dense1, torch::Tensor dense2, torch::Tensor Ap, torch::Tensor Aj, int d1_n_row, int d1_n_col, int d2_n_col){
    CHECK_INPUT(dense1);
    CHECK_INPUT(dense2);
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    return smo_d_d_s_mm_cuda(dense1, dense2, Ap, Aj, d1_n_row, d1_n_col, d2_n_col);
}

torch::Tensor smo_d_s_d_mm(torch::Tensor dense_t, torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int d_t_n_row, int d_t_n_col, int s_n_col){
    CHECK_INPUT(dense_t);
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    CHECK_INPUT(Ax);
    return smo_d_s_d_mm_cuda(dense_t, Ap, Aj, Ax, d_t_n_row, d_t_n_col, s_n_col);
}

torch::Tensor smo_s_d_s_add(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, torch::Tensor dense_vector, int s_n_row, int s_n_col){
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    CHECK_INPUT(Ax);
    CHECK_INPUT(dense_vector);
    return smo_s_d_s_add_cuda(Ap, Aj, Ax, dense_vector, s_n_row, s_n_col);
}

torch::Tensor smo_s_sum(torch::Tensor Ap, torch::Tensor Aj, torch::Tensor Ax, int n_row, int n_col){
    CHECK_INPUT(Ap);
    CHECK_INPUT(Aj);
    CHECK_INPUT(Ax);
    return smo_s_sum_cuda(Ap, Aj, Ax, n_row, n_col);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smo_s_transpose", &smo_s_transpose, "sparse matrix transpose");
  m.def("smo_d_transpose", &smo_d_transpose, "dense matrix transpose");
  m.def("smo_s_d_d_mm", &smo_s_d_d_mm, "sparse * dense = dense");
  m.def("smo_d_d_s_mm", &smo_d_d_s_mm, "dense * dense = sparse");
  m.def("smo_d_s_d_mm", &smo_d_s_d_mm, "dense * sparse = dense");
  m.def("smo_s_d_s_add", &smo_s_d_s_add, "sparse + dense vector = sparse");
  m.def("smo_s_sum", &smo_s_sum, "sparse matrix sum (dim = 0)");
}