import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from ..utils.smo import *
from .InputLayer import InputLayer
from .OutputLayer import OutputLayer
from TLGCN.utils import globalData
import numpy as np
import random
from torch.nn import init

import math
device = torch.device("cuda")
def activation(input, kind):
    if kind == 'selu':
        return torch.selu(input)
    elif kind == 'relu':
        return torch.relu(input)
    elif kind == 'relu6':
        return torch.relu6(input)
    elif kind == 'sigmoid':
        return torch.sigmoid(input)
    elif kind == 'tanh':
        return torch.tanh(input)
    elif kind == 'elu':
        return torch.elu(input)
    elif kind == 'lrelu':
        return torch.leaky_relu(input)
    elif kind == 'swish':
        return input * torch.sigmoid(input)
    elif kind == 'none':
        return input
    else:
        raise ValueError('Unknown non-linearity type')


def eval(input, target):
    temp_RMSE = torch.sum((target - input)**2).item()
    temp_MAE = torch.sum(torch.abs(target - input)).item()
    return temp_RMSE, temp_MAE


def MSEloss(input, target):
    """

    :param input: the model's output
    :param target: the target label
    :return:
    """
    pass

def seed_torch(seed=1021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # init.xavier_normal_(m.weight.data)
        # m.weight.data.normal_(0,0.01)
        init.normal_(m.weight.data, std=0.1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        init.xavier_normal_(m.weight.data)
        # init.uniform_(m.weight.data, a=0, b=0.004)
        # init.normal_(m.weight.data, std=0.1)

class Fast_AutoEncoder(nn.Module):
    def __init__(self, layer_sizes, drop_prob, lr, num_items, graph, num_users, p, n_layers):
        """
        a fast autoencoder

        :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
            For example: [10000, 1024, 512] will result in:
            - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
            - decoder 2 layers: 512x1024 and 1024x10000.
        :param layer_type:  (default 'selu') Type of no-linearity
        :param dp_drop_prob: (default: 0.0) Dropout drop probability
        """
        super(Fast_AutoEncoder, self).__init__()
        self._layer_sizes = layer_sizes
        self._drop_prob = drop_prob
        self.graph = graph
        self.num_users = num_users
        self.num_items = num_items
        self.p = p
        self.n_layers = n_layers

        seed_torch()
        self.embedding_item = nn.Embedding(num_embeddings=num_items, embedding_dim=layer_sizes[1]).apply(
            weigth_init).cuda()
        self.embedding_user = nn.Embedding(num_embeddings=num_users, embedding_dim=layer_sizes[1]).apply(
            weigth_init).cuda()
        self.embedding_item_cache = nn.Embedding(num_embeddings=num_items, embedding_dim=layer_sizes[1]).apply(
            weigth_init).cuda()
        self.weight_gcn = nn.Parameter(torch.rand((layer_sizes[len(layer_sizes) - 1], layer_sizes[1])))
        print("--------test-----")
        print(self.embedding_item.weight[:512,].shape)
        print("p: ", self.p)

        self.encoder_w = nn.ParameterList(
            [nn.Parameter(torch.rand((self._layer_sizes[i], self._layer_sizes[i + 1]))) for i in range(len(self._layer_sizes) - 1)])
        for ind, w in enumerate(self.encoder_w):
            weight_init.xavier_normal_(w)
            # weight_init.uniform_(w, a=0, b=0.0004)
        self.encoder_b = nn.ParameterList([nn.Parameter(torch.zeros(self._layer_sizes[i + 1]))
                                           for i in range(len(self._layer_sizes) - 1)])

        self.shift = 2
        self.layer_num = globalData.get("layer_num")

        # two hidden layers
        if self.layer_num % 2 == 0:
            reversed_enc_layers = list(reversed(self._layer_sizes))[1:]
        else:
            reversed_enc_layers = list(reversed(self._layer_sizes))
        self.decoder_w = nn.ParameterList([nn.Parameter(torch.rand(
            reversed_enc_layers[i], reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

        for ind, w in enumerate(self.decoder_w):
            print(w.shape, ind)
            if ind == len(self.decoder_w) - 1:
                # weight_init.uniform_(w, a = 0, b = 0.004)
                # print(w)
                # print(w)
                # print(torch.min(torch.sigmoid(w)))
                # weight_init.xavier_uniform_(w)
                weight_init.xavier_normal_(w)
            else:
                weight_init.xavier_normal_(w)
            # print(torch.min(w))
        self.decoder_b = nn.ParameterList([nn.Parameter(torch.zeros(reversed_enc_layers[i + 1]))
                                           for i in range(len(reversed_enc_layers) - 1)])

    def encoder(self, Ap, Aj, Ax, Flag, isTrain):
        for ind, w in enumerate(self.encoder_w):
            if ind == 0:
                if Flag:
                    x = InputLayer.apply(Ap, Aj, Ax, w, self.encoder_b[ind], self._drop_prob)
                    self.first_x = x
                    self.global_x = x
                    # print("there is a flase")
                else:
                    x = InputLayer.apply(Ap, Aj, Ax, w, self.encoder_b[ind], 0.0)
                    self.first_x = x
                    self.global_x = x
            else:
                x = activation(input = F.linear(x, w.t(), self.encoder_b[ind]).to(device), kind="sigmoid")
                x = F.dropout(x, p=self._drop_prob, training=isTrain)
                if ind==1:
                    # x = self.first_x + x
                    self.second_x = x
                else:
                    self.third_x = x

        return x

    def decoder(self, Ap, Aj, Ax, x, isTrain):
        for ind, w in enumerate(self.decoder_w):
            # decoder odd -3 normal: -2
            # two hidden layers
            if self.layer_num % 2 == 0:
                self.shift = 3
            if ind == len(self._layer_sizes) - self.shift:
                p = self.p
                q = 1 - p
                # concated embeddings of users and items: first layer E
                all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])

                # all layers embs, here adding the first layer
                all_emb = all_emb
                embs = [all_emb]

                # calculate the remaining layers embs
                for layer in range(self.n_layers):
                    all_emb = torch.sparse.mm(self.graph, all_emb)
                    embs.append(all_emb)

                # # concatenate all the embs
                embs = torch.stack(embs, dim=1)

                # # print(embs.size())
                # # calculate the average embs and split it to users and items
                light_out = torch.mean(embs, dim=1)

                w, items_avg  = torch.split(light_out, [self.num_users, self.num_items])

                w = w.t().contiguous()
                items_avg = items_avg.contiguous()

                # print(w.shape, users_avg.shape)
                if isTrain:
                    x = p * items_avg[
                            globalData.get("idx") - x.shape[0]:globalData.get("idx"), ] + q * x

                    x = OutputLayer.apply(x, w, self.decoder_b[ind], Ap, Aj, Ax)
                else:
                    x = p * items_avg[
                            globalData.get("idx") - x.shape[0]:globalData.get("idx"), ] + q * x

                    x = torch.mm(x, w) + self.decoder_b[ind]
            else:
                x = activation(input=F.linear(x, w.t(), self.decoder_b[ind]), kind='sigmoid')
                x = F.dropout(x, p=self._drop_prob, training=isTrain)
                if self.layer_num == 5:
                    if ind == 0:
                        self.forth_x = x

        return x

    def forward(self, Ap, Aj, Ax, Flag, isTrain):
        z = self.encoder(Ap, Aj, Ax, Flag, isTrain)
        y = self.decoder(Ap, Aj, Ax, z, isTrain)
        return y
