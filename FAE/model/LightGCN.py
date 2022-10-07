import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import random
import math

def seed_torch(seed=1029):
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
        # init.xavier_normal(m.weight.data)
        init.normal_(m.weight.data, std=0.1)
        # m.weight.data.normal_(0,0.01)
        # m.bias.data.zero_()

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, n_layers, graph):
        """
        a neural CF model

        :param num_users and num_items
        :param factors-LF size
        :param hidden_layers [layer1_size, layer2_size, ......]
        """
        super(LightGCN, self).__init__()
        seed_torch()

        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.graph = graph

        # embedding layer
        self.embedding_user = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim).apply(
            weigth_init).cuda()
        self.embedding_item = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim).apply(
            weigth_init).cuda()

    def forward(self, user_input, item_input):
        # first layer embeddings, also the unique parameters
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # concated embeddings of users and items: first layer E
        all_emb = torch.cat([users_emb, items_emb])

        # all layers embs, here adding the first layer
        embs = [all_emb]

        # calculate the remaining layers embs
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)

        # concatenate all the embs
        embs = torch.stack(embs, dim=1)

        # print(embs.size())
        # calculate the average embs and split it to users and items
        light_out = torch.mean(embs, dim=1)
        users_avg, items_avg = torch.split(light_out, [self.num_users, self.num_items])

        # extract the embedings of inputed users and items
        users_emb = users_avg[user_input]
        items_emb = items_avg[item_input]

        # calculate the preds
        inner_pro = torch.mul(users_emb, items_emb)
        preds = torch.sum(inner_pro, dim=1)

        return preds

if __name__ == '__main__':
    NCF_MLP = NCF_MLP(num_users=10, num_items=5, fators=64, hidden_layers=[64, 16])
    user_input = torch.LongTensor([0, 1, 2])
    item_input = torch.LongTensor([0, 1, 2])
    pred = NCF_MLP.forward(user_input=user_input, item_input=item_input)
    print(pred.size())
    print(pred)
    # print(NCF_MLP.MLP_Embedding_User.weight, NCF_MLP.MLP_Embedding_User.weight)
    # print(concatenation_embeddings.size())
    # print(concatenation_embeddings)
