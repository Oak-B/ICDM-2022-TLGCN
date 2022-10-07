import torch
from torch.autograd import Function
from torch.autograd import Variable
from ..utils.smo import *
from FAE.utils import globalData


class OutputLayer(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, Ap, Aj, Ax):
        # ctx.save_for_backward(input, weight, bias, Ap, Aj)
        # # use sigmoid to control the nonnegativity
        update_manner = globalData.get("update_manner")
        if update_manner == "non_relu":
            weight = torch.relu(weight)
        elif update_manner == "non_sigmoid":
            weight = torch.sigmoid(weight)

        output = smo_d_d_s_mm(input, weight, Ap, Aj, input.size(0), input.size(1), weight.size(1))
        output = smo_s_d_s_add(Ap, Aj, output, bias, Ap.size(0) - 1, weight.size(1))

        ctx.save_for_backward(input, weight, bias, Ap, Aj, Ax, output)
        # print(torch.max(lr_weight_adp_nn))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, Ap, Aj, Ax, output = ctx.saved_tensors
        grad_input = smo_s_d_d_mm(
            Ap,
            Aj,
            grad_output,
            smo_d_transpose(
                weight,
                weight.size(0),
                weight.size(1)),
            Ap.size(0) - 1,
            weight.size(0))
        grad_weight = smo_d_s_d_mm(input.t(), Ap, Aj, grad_output, input.size(1), input.size(0), weight.size(1))
        grad_bias = smo_s_sum(Ap, Aj, grad_output, Ap.size(0)-1, weight.size(1))

        '''
        positive and negative term decomposition of gradient
        '''
        grad_output_alpha = 2 * output
        grad_output_beta = 2 * Ax

        grad_weight_alpha = smo_d_s_d_mm(input.t(), Ap, Aj, grad_output_alpha, input.size(1), input.size(0), weight.size(1))
        grad_weight_beta = smo_d_s_d_mm(input.t(), Ap, Aj, grad_output_beta, input.size(1), input.size(0), weight.size(1))

        # print("outputLayer: ", torch.min(grad_weight_alpha))
        # if sigmoid
        # ones = torch.ones(weight.size(0), weight.size(1)).cuda()
        # grad_sigmoid_weight = torch.mul(torch.add(ones, -weight), weight) / 5
        # false grad_relu
        # grad_sigmoid_weight = torch.where(torch.gt(weight, 0),
        #                                  torch.full_like(weight, 1),
        #                                  weight)
        # grad_weight_alpha = torch.mul(grad_weight_alpha, grad_sigmoid_weight)
        # grad_weight_beta = torch.mul(grad_weight_beta, grad_sigmoid_weight)
        update_manner = globalData.get("update_manner")
        if update_manner == "non_relu":
            grad_relu_weight = torch.where(torch.gt(weight, 0),
                                           torch.full_like(weight, 1),
                                           weight)
            grad_weight_alpha = torch.mul(grad_weight_alpha, grad_relu_weight)
            grad_weight_beta = torch.mul(grad_weight_beta, grad_relu_weight)
        elif update_manner == "non_sigmoid":
            ones = torch.ones(weight.size(0), weight.size(1)).cuda()
            grad_sigmoid_weight = torch.mul(torch.add(ones, -weight), weight)
            grad_weight_alpha = torch.mul(grad_weight_alpha, grad_sigmoid_weight)
            grad_weight_beta = torch.mul(grad_weight_beta, grad_sigmoid_weight)

        grad_bias_alpha = smo_s_sum(Ap, Aj, grad_output_alpha, Ap.size(0)-1, weight.size(1))
        grad_bias_beta = smo_s_sum(Ap, Aj, grad_output_beta, Ap.size(0) - 1, weight.size(1))

        globalData.set("grad_weight_alpha", grad_weight_alpha)
        globalData.set("grad_weight_beta", grad_weight_beta)
        globalData.set("grad_bias_alpha", grad_bias_alpha)
        globalData.set("grad_bias_beta", grad_bias_beta)

        grad_weight1 = grad_weight_alpha - grad_weight_beta
        grad_bias1 = grad_bias_alpha - grad_bias_beta
        '''
        test
        '''
        # max_batch_lr = torch.max(lr_weight_adp_nn)
        # max_lr = globalData.get("max_lr")
        # max_lr = max(max_lr,max_batch_lr)
        # globalData.set("max_lr",max_lr)
        #
        # print(torch.max(lr_weight_adp_nn))
        # print(grad_weight1[0][0],grad_weight_alpha[0][0],lr_weight_adp_nn[0][0])
        # print(grad_weight1[0][0], grad_weight[0][0])
        # print(grad_input)
        # print(grad_output)
        # print(grad_weight)
        # print(grad_bias)
        return grad_input, grad_weight, grad_bias, None, None, None
