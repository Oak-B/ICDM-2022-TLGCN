import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn as nn
from ..utils.smo import *


class InputLayer(Function):
    @staticmethod
    def forward(ctx, Ap, Aj, Ax, weight, bias, drop_prob):
        """
        the input layer's forward

        :param ctx: ctx is a context object that can be used to stash information
        :param Ap: row pointer of sparse matrix
        :param Aj: column indices of sparse matrix
        :param Ax: nonzeros of sparse matrix
        :param weight: the weight
        :param bias: the bias
        :return: the output tensor
        """

        output = torch.sigmoid(smo_s_d_d_mm(Ap, Aj, Ax, weight, Ap.size(0) - 1, weight.size(1)) + bias)
        drop_mask = torch.full((output.size(0), output.size(1)), 1)
        if drop_prob > 0.0:
            drop_mask = torch.bernoulli(torch.full((output.size(0),output.size(1)),
                                drop_prob,
                                device = torch.device('cuda:1')))
            output.mul_(drop_mask)
            output /= (1 - drop_prob)

        drop_prob = torch.tensor([drop_prob])
        par_output = output * (1 - output)
        ctx.save_for_backward(Ap, Aj, Ax, weight, bias, par_output, drop_prob, drop_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Ap, Aj, Ax, weight, bias, par_output, drop_prob , drop_mask = ctx.saved_tensors

        drop_prob = drop_prob.item()
        if drop_prob > 0.0:
            grad_output.mul_(drop_mask)
            grad_output /= (1-drop_prob)

        Ap_t, Aj_t, Ax_t = smo_s_transpose(Ap, Aj, Ax, Ap.size(0) - 1, weight.size(0))
        grad_output.mul_(par_output)
        grad_weight = smo_s_d_d_mm(Ap_t, Aj_t, Ax_t, grad_output, weight.size(0), weight.size(1))
        grad_bias = grad_output.sum(0)
        #print(grad_weight)
        return None, None, None, grad_weight, grad_bias, None
