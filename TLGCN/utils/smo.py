"""
sparse matrix operations.\n
All operations are on GPU.
"""

import torch
from TLGCN.utils.jit import smo


def smo_s_transpose(Ap, Aj, Ax, n_row, n_col):
    """
    sparse matrix transpose

    :param Ap: row pointer
    :param Aj: column indices
    :param Ax: nonzeros
    :param n_row: number of rows
    :param n_col: number of columns
    :return: the transpose sparse matirx
    """
    return smo.smo_s_transpose(Ap, Aj, Ax, n_row, n_col)


def smo_d_transpose(X, H, W):
    """
    dense matrix transpose

    :param X: the dense matrix
    :param H: the number of rows
    :param W: the number of cols
    :return: the transpose dense matrix
    """
    return smo.smo_d_transpose(X, H, W)


def smo_s_d_d_mm(Ap, Aj, Ax, dense, n_row, n_col):
    """
    sparse * dense =dense

    :param Ap: row pointer of sparse matrix
    :param Aj: column indices of sparse matrix
    :param Ax: nonzeros of sparse matrix
    :param dense: dense matrix
    :param n_row: number of rows of sparse matrix
    :param n_col: number of columns of dense matrix
    :return:
    """
    return smo.smo_s_d_d_mm(Ap, Aj, Ax, dense, n_row, n_col)


def smo_d_d_s_mm(dense1, dense2, Ap, Aj, d1_n_row, d1_n_col, d2_n_col):
    """
    dense * dense = sparse

    :param dense1: the first dense matrix
    :param dense2: the second dense matrix
    :param Ap: row pointer of the output sparse matrix
    :param Aj: column indices of the output sparse matrix
    :param d1_n_row: number of rows of the first dense matrix
    :param d1_n_col: number of cols of the first dense matrix
    :param d2_n_col: number of cols of the second dense matrix
    :return: nonzeors of the output sparse matrix
    """
    return smo.smo_d_d_s_mm(dense1, dense2, Ap, Aj, d1_n_row, d1_n_col, d2_n_col)


def smo_d_s_d_mm(dense, Ap, Aj, Ax, d_n_row, d_n_col, s_n_col):
    """
    dense * sparse = dense

    :param dense: the dense matrix
    :param Ap: row pointer of the sparse matrix
    :param Aj: column indices of the sparse matrix
    :param Ax: nonzeros of the sparse matrix
    :param d_n_row: number of rows of the dense matrix
    :param d_n_col: number of cols of the dense matrix
    :param s_n_col: number of cols of the sparse matrix
    :return: the output dense matrix
    """
    if dense.t().is_contiguous():
        return smo.smo_d_s_d_mm(dense.t(), Ap, Aj, Ax, d_n_col, d_n_row, s_n_col)
    else:
        return smo.smo_d_s_d_mm(dense.t().contiguous(), Ap, Aj, Ax, d_n_col, d_n_row, s_n_col)


def smo_s_d_s_add(Ap, Aj, Ax, dense_vector, s_n_row, s_n_col):
    """
    sparse + dense vector = sparse

    :param Ap: row pointer of the sparse matrix
    :param Aj: column indices of the sparse matrix
    :param Ax: nonzeros of the sparse matrix
    :param dense_vector: dense vector
    :param s_n_row: number of rows of the sparse matrix
    :param s_n_col: number of cols of the sparse matrix
    :return: the output sparse matrix
    """
    return smo.smo_s_d_s_add(Ap, Aj, Ax, dense_vector, s_n_row, s_n_col)


def smo_s_sum(Ap, Aj, Ax, n_row, n_col):
    """
    compute the spare tensor's sum.
    Default the dim is 0.

    :param Ap: row pointer of the sparse matrix
    :param Aj: column indices of the sparse matrix
    :param Ax: nonzeros of the sparse matrix
    :param n_row: number of rows of the sparse matrix
    :param n_col: number of cols of the sprse matrix
    :return:
    """
    return smo.smo_s_sum(Ap, Aj, Ax, n_row, n_col)
