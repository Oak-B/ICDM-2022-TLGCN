import torch
from torch.autograd import Function
import torch.nn as nn

class MSE(Function):
    @staticmethod
    def forward(ctx, input, target):
        """

        :param input: the model's output
        :param target: the target label
        :return:
        """
        ctx.save_for_backward(input, target)
        #output = torch.sum(torch.pow((target - input), 2)) + L2/2*(torch.norm(para[0]) + torch.norm(para[1]) + torch.norm(para[2]) + torch.norm(para[3]))
        output = torch.sum(torch.pow((target - input), 2))
        return output

    @staticmethod
    def backward(ctx, grad_output=None):
        input, target = ctx.saved_tensors
        grad_input = 2 * (input - target)
        # print(input[0],target[0])
        grad_target = None
        return grad_input, grad_target
