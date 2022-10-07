import torch
import time
import math
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.init as weight_init
from TLGCN.model.FAE import Fast_AutoEncoder
from TLGCN.model.FAE import eval
from TLGCN.model.FAE import MSEloss
from torch.autograd import Variable
from TLGCN.utils.LoadData import Data
from TLGCN.utils.smo import *
from TLGCN.model.MSE import MSE
from TLGCN.utils import globalData
import os
import warnings

warnings.filterwarnings('ignore')
torch.cuda.set_device(0)

min_RMSE_ALL = 10000
min_MAE_ALL = 10000


def main(LR, L2, batch_size, size, mom, drop_prob, name, update_manner, optimizer_type, val_or_test, layer_num, idx, p, n_layers):
    global min_RMSE_ALL
    globalData.init()
    globalData.set("max_lr", LR)
    globalData.set("min_weight", 1000)
    globalData.set("min_bias", 1000)
    globalData.set("update_manner", update_manner)
    globalData.set("layer_num", layer_num)
    num_epochs = 1000
    layer_sizes = size
    data = Data("../data/" + name + "/train.txt", "../data/" + name + "/" + val_or_test + ".txt", n_layers, batch_size)

    fast_autoencoder = Fast_AutoEncoder(layer_sizes=[data.max_user_id] + size,
                                        drop_prob=drop_prob, lr = LR, num_items=data.max_item_id, graph = data.Graph,
                                        num_users = data.max_user_id, p = p, n_layers=n_layers)

    fast_autoencoder = fast_autoencoder.cuda()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(fast_autoencoder.parameters(), lr=LR, weight_decay=L2)
        if update_manner == "normal":
            save_dir = "result/SGD/" + name + "/"
        else:
            save_dir = "result/NSGD/" + name + "/"
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(fast_autoencoder.parameters(), lr=LR, weight_decay=L2)
        if update_manner == "normal":
            save_dir = "result/Adam/" + name + "/"
        else:
            save_dir = "result/NAdam/" + name + "/"
    elif optimizer_type == "NAdam":
        optimizer = optim.Adam(fast_autoencoder.parameters(), lr=LR, weight_decay=L2)
        if update_manner == "normal":
            save_dir = "result/Adam/" + name + "/"
        else:
            save_dir = "result/NAdam/" + name + "/"
    else:
        raise ValueError('Unknown optimizer_type.')

    result_file = open(save_dir + name
                       + "_" + update_manner
                       + "_" + val_or_test
                       + "_" + str(layer_num)
                       + "_learningRate=" + str(LR)
                       + "_eta=" + str(L2)
                       + "_hiddenSize=" + str(size)
                       + "_idx=" + str(idx)
                       + "_p=" + str(p)
                       + "_gcnlayers=" + str(n_layers)
                       + "_batchSize=" + str(batch_size)
                       + ".txt", "w")

    print(name + "\n"
          + "update_manner=" + update_manner + "\n"
          + "val_or_test=" + val_or_test + "\n"
          + "layer_num=" + str(layer_num) + "\n"
          + "learningRate=" + str(LR) + "\n"
          + "eta=" + str(L2) + "\n"
          + "idx=" + str(idx) + "\n"
          + "hiddenSize=" + str(size) + "\n"
          + "p=" + str(p) + "\n"
          + "n_layers=" + str(n_layers) + "\n"
          + "batchSize=" + str(batch_size))

    min_RMSE = 10000
    min_MAE = 10000
    min_index_mae = 0
    all_start_time = time.time()
    total_start_time = time.time()
    for i in range(num_epochs):
        # train
        drop_flag = True
        isTrain = True
        # switch to train mode
        fast_autoencoder.train()
        torch.cuda.synchronize()
        start_time = time.time()

        # for name, param in fast_autoencoder.named_parameters():
        #     print(name)

        idx = 0
        for mb in data.train_iterate_one_epoch():
            # torch.cuda.synchronize()
            # row,coloum,rate
            idx += mb.shape[0]
            globalData.set("idx", idx)
            Ap = Variable(torch.IntTensor(mb.indptr).cuda())
            Aj = Variable(torch.IntTensor(mb.indices).cuda())
            Ax = Variable(torch.FloatTensor(mb.data).cuda())
            # calculate the col-correponding size
            Ap_t, Aj_t, Ax_t = smo_s_transpose(Ap, Aj, Ax, Ap.size(0) - 1, data.max_user_id)
            each_user_corres_items_size = torch.add(Ap_t[1:], -Ap_t[:len(Ap_t) - 1])
            batch_elements_size = Ap_t[len(Ap_t) - 1]
            globalData.set("each_user_corres_items_size", each_user_corres_items_size)
            globalData.set("batch_elements_size", batch_elements_size)

            optimizer.zero_grad()
            outputs = fast_autoencoder.forward(Ap, Aj, Ax, drop_flag, isTrain)
            #outputs = nn.ReLU()(outputs)
            loss = MSE.apply(outputs, Ax)

            loss.backward()
            optimizer.step()
            # fast_autoencoder.embedding_item.weight = fast_autoencoder.embedding_item_cache.weight
        print(globalData.get("min_weight"), globalData.get("min_bias"))

        torch.cuda.synchronize()
        end_time = time.time()

        # test
        RMSE = 0.0
        MAE = 0.0
        num = 0
        drop_flag = False
        isTrain = False
        idx = 0
        for mb_train, mb_test in data.test_iterate_one_epoch():
            idx += mb_train.shape[0]
            globalData.set("idx", idx)
            Ap = Variable(torch.IntTensor(mb_train.indptr).cuda())
            Aj = Variable(torch.IntTensor(mb_train.indices).cuda())
            Ax = Variable(torch.FloatTensor(mb_train.data).cuda())

            outputs = fast_autoencoder(Ap, Aj, Ax, drop_flag, isTrain)
            target = Variable(torch.FloatTensor(mb_test.todense()).cuda())
            mask = torch.sign(target)
            outputs = outputs * mask
            #outputs = nn.ReLU()(outputs)

            temp_RMSE, temp_MAE = eval(outputs, target)
            temp_tol_num = torch.nonzero(target).size(0)
            RMSE += temp_RMSE
            MAE += temp_MAE
            num += temp_tol_num
        RMSE = math.sqrt(RMSE / num)
        MAE = MAE / num

        print(str(i) + "\t" + str(RMSE) + "\t" + str(MAE) + "\t" + str(end_time - start_time))
        result_file.write(str(i) + "\t" + str(RMSE) + "\t" + str(MAE) + "\t" + str(end_time - start_time) + "\n")
        result_file.flush()
        if RMSE < min_RMSE:
            min_RMSE = RMSE
            min_index = i + 1
            all_end_time_RMSE = time.time()
        if MAE < min_MAE:
            min_MAE = MAE
            min_index_mae = i + 1
            all_end_time_MAE = time.time()
        if min_RMSE < min_RMSE_ALL:
            min_RMSE_ALL = min_RMSE
        if i - min_index > 30 and i - min_index_mae > 30:
            all_time_minRMSE = all_end_time_RMSE - all_start_time
            all_time_minMAE = all_end_time_MAE - all_start_time
            per_time_avg_RMSE = all_time_minRMSE / min_index
            per_time_avg_MAE = all_time_minMAE / min_index_mae
            print('min_RMSE: ' + str(min_RMSE))
            print('min_MAE: ' + str(min_MAE))
            print("all_time_minRMSE: " + str(all_time_minRMSE))
            print("all_time_minMAE: " + str(all_time_minMAE))
            print("all_iteration_count_minRMSE: " + str(min_index))
            print("all_iteration_count_minMAE: " + str(min_index_mae))
            print("per_time_avg_RMSE: " + str(per_time_avg_RMSE))
            print("per_time_avg_MAE: " + str(per_time_avg_MAE))
            result_file.write("min_RMSE: " + str(min_RMSE) + "\n")
            result_file.write("min_MAE: " + str(min_MAE) + "\n")
            result_file.write("all_time_minRMSE: " + str(all_time_minRMSE) + "\n")
            result_file.write("all_time_minMAE: " + str(all_time_minMAE) + "\n")
            result_file.write("all_iteration_count_minRMSE: " + str(min_index) + "\n")
            result_file.write("all_iteration_count_minMAE: " + str(min_index_mae) + "\n")
            result_file.write("per_time_avg_RMSE: " + str(per_time_avg_RMSE) + "\n")
            result_file.write("per_time_avg_MAE: " + str(per_time_avg_MAE) + "\n")
            break
    total_end_time = time.time()
    print("total time: ", total_end_time - total_start_time)
    result_file.write("total time: " + str(total_end_time - total_start_time))
    result_file.close()


if __name__ == '__main__':
    torch.cuda.set_device(3)
    for idx in [20220913]:
        for name in ["rt5_t1v4"]:
            for p in [0.2]:
                for lr in [5e-3]:
                    for layer_dim in [128]:
                        for layer_num_fully in [3]:
                            for layer_num_graph in [5]:
                                main(lr, 0.1, 512, [layer_dim, layer_dim], 0.0, 0.0, name, "normal", "Adam", "test", layer_num_fully, idx, p, layer_num_graph)
                                #main(lr, 0.1, 512, [layer_dim], 0.0, 0.0, name, "normal", "Adam", "val", 1, idx, p, layer_num_graph)
                                #main(lr, 0.1, 512, [layer_dim, layer_dim], 0.0, 0.0, name, "normal", "Adam", "val", 3, idx, p, layer_num_graph)
                                #main(lr, 0.1, 512, [layer_dim, layer_dim, layer_dim], 0.0, 0.0, name, "normal", "Adam", "val", 4, idx, p, layer_num_graph)
                                #main(lr, 0.1, 512, [layer_dim, layer_dim, layer_dim], 0.0, 0.0, name, "normal", "Adam", "val", 5, idx, p, layer_num_graph)


