from random import shuffle
from scipy.sparse import csr_matrix
import numpy as np
import warnings
import torch
import scipy.sparse as sp
import time

warnings.filterwarnings("ignore")
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
class Data:
    def __init__(self, train_file, test_file, n_layers, batch_size=256):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.delimiter = ","
        self.max_user_id = -1
        self.max_item_id = -1
        self.n_layers = n_layers

        print("start loading data...")

        train_u = []
        train_i = []
        train_rating = []
        with open(self.train_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(self.delimiter)
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                self.max_user_id = user_id if user_id > self.max_user_id else self.max_user_id
                self.max_item_id = item_id if item_id > self.max_item_id else self.max_item_id
                train_u.append(user_id - 1)
                train_i.append(item_id - 1)
                train_rating.append(rating)
        test_u = []
        test_i = []
        test_rating = []
        with open(self.test_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(self.delimiter)
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                self.max_user_id = user_id if user_id > self.max_user_id else self.max_user_id
                self.max_item_id = item_id if item_id > self.max_item_id else self.max_item_id
                test_u.append(user_id - 1)
                test_i.append(item_id - 1)
                test_rating.append(rating)
        self.train_data = csr_matrix(
            (train_rating, (train_i, train_u)), shape=(
                self.max_item_id, self.max_user_id), dtype='float32')
        self.test_data = csr_matrix(
            (test_rating, (test_i, test_u)), shape=(
                self.max_item_id, self.max_user_id), dtype='float32')
        # (users,items), bipartite graph: sparse R, which is the adjacent matrix of UI
        # self.UserItemNet = csr_matrix(
        #         (train_rating, (train_u, train_i)), shape=(
        #             self.max_user_id, self.max_item_id), dtype='float32')
        self.UserItemNet = csr_matrix(
            (np.ones(len(train_u)), (train_u, train_i)), shape=(
                self.max_user_id, self.max_item_id), dtype='float32')
        # self.UserItemNet = csr_matrix(
        #     (np.ones(len(train_u) + len(test_u)),
        #      (np.hstack((train_u, test_u)), np.hstack((train_i, test_i)))),
        #     shape=(self.max_user_id, self.max_item_id), dtype='float32')
        print("loading data done...")
        print("Number of users:", self.max_user_id, "Number of items:", self.max_item_id)
        # print(self.train_data)
        # print(self.UserItemNet)
        self.build_graph()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def build_graph(self):
        print("loading adjacency matrix...")
        try:
            pre_adj_mat = sp.load_npz('pre_adj_mat/' + self.dataset_name + '_s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except:
            print("there is not be a generated adj_mat, generating adjacency matrix...")
            s = time.time()

            R_coo = self.UserItemNet.tocoo()
            R_row = R_coo.row
            R_col = R_coo.col + self.max_user_id
            R_data = R_coo.data
            adj_row = np.hstack((R_col, R_row))
            adj_col = np.hstack((R_row, R_col))
            adj_data = np.hstack((R_data, R_data))
            adj_coo = sp.coo_matrix((adj_data, (adj_row, adj_col)))

            print("adj_coo_size: ", adj_coo.shape)
            adj_mat = adj_coo.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            print('costing ' + str(end - s) + 's, saved norm_mat...')
            # sp.save_npz('pre_adj_mat/' + self.dataset_name + '_s_pre_adj_mat.npz', norm_adj)
            print("generating adjacency matrix done...")

        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)

    def train_iterate_one_epoch(self):
        idxs = np.random.permutation(self.max_item_id)
        s_ind = 0
        e_ind = self.batch_size
        while s_ind < self.max_item_id:
            batch_idxs = idxs[s_ind:min(e_ind, self.max_item_id)]
            batch_data = self.train_data[batch_idxs]
            s_ind += self.batch_size
            e_ind += self.batch_size
            yield batch_data

    def test_iterate_one_epoch(self):
        idxs = np.unique(self.test_data.tocoo().row)
        s_ind = 0
        e_ind = self.batch_size
        while s_ind < len(idxs):
            batch_idxs = idxs[s_ind:min(e_ind, len(idxs))]
            batch_data_train = self.train_data[batch_idxs]
            batch_data_test = self.test_data[batch_idxs]
            s_ind += self.batch_size
            e_ind += self.batch_size
            yield batch_data_train, batch_data_test
