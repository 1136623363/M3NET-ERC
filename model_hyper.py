import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
# from hyper import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
# from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv
from sklearn.cluster import KMeans
import warnings
from utils import *

# 忽略FutureWarning类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
# 忽略特定警告
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL*")

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp + i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx, a], dim=0)
            tmp = tmp + i
        # x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)


class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant,
                 return_feature, use_residue,
                 new_graph='full', n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3,
                 num_K=4,windows=5,step=6):
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  # True
        self.use_residue = use_residue
        self.new_graph = new_graph

        # self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
        #                       dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
        #                       return_feature=return_feature, use_residue=use_residue)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.l_pos = PositionalEncoding(n_dim)
        self.use_position = False
        # ------------------------------------
        self.fc1 = nn.Linear(n_dim, nhidden)
        # self.fc2 = nn.Linear(n_dim, nhidden)
        self.num_L = num_L
        self.num_K = num_K
        self.windows = windows
        self.step = step
        self.beita = nn.Parameter(torch.Tensor([0.5]))
        # self.hyperedge_weight = []
        # self.EW_weight = []
        for ll in range(num_L):
            setattr(self, 'hyperconv%d' % (ll + 1), HypergraphConv(nhidden, nhidden))

            # self.hyperedge_weight.append(nn.Parameter(torch.ones(20000)))
            # self.EW_weight.append(nn.Parameter(torch.ones(20000)))
        self.act_fn = nn.ReLU()

        # self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        # self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_weight = nn.Parameter(torch.ones(20000))
        self.EW_weight = nn.Parameter(torch.ones(20000))
        # self.hyperedge_weight = nn.Parameter(torch.rand(10000))
        # self.EW_weight = nn.Parameter(torch.rand(10000))

        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        # nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self, 'conv%d' % (kk + 1), highConv(nhidden, nhidden))
        # self.conv = highConv(nhidden, nhidden)

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
                #只在l文本模态加入说话者嵌入
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.l_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.l_pos(v, dia_len)
        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len,
                                                                                                self.modals)
        # hyperedge_index, (2,节点数x2) (点,超边) 有点类似节点与超边的关联矩阵  第一行是节点，第二行是超边 [nodeID, hyperedgeID]
        # edge_index,  (2,普通边数)
        # features,  (节点个数,节点特征)
        # batch,    (节点个数,)
        # hyperedge_type1 超边类型,分两种,0同模态超边,1跨模态超边   len为超边个数:3*batch_size+N


        # hyperedge_index3 = self.create_hyper_index2(a, v, l, dia_len,self.modals,int(hyperedge_index.data.cpu()[-1][-1]))
        # hyperedge_index3 = self.create_hyper_index5(a, v, l, dia_len,self.modals,int(hyperedge_index.data.cpu()[-1][-1]))
        hyperedge_index2 = self.create_hyper_index7(a, v, l, dia_len, self.modals,int(hyperedge_index.data.cpu()[-1][-1])+1,spk_idx)
        hyperedge_index3 = self.create_hyper_index6(a, v, l, dia_len, self.modals,int(hyperedge_index2.data.cpu()[-1][-1])+1,spk_idx,self.windows,self.step)
        # hyperedge_index4 = self.create_hyper_index8(a, v, l, dia_len, self.modals,int(hyperedge_index3.data.cpu()[-1][-1])+1, spk_idx, self.windows,self.step)
        #
        hyperedge_index = torch.cat([hyperedge_index,hyperedge_index2,hyperedge_index3], dim=1)
        """
        """

        x1 = self.fc1(features)
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]
        # print(weight[:5], EW_weight)
        edge_attr = self.hyperedge_attr1 * hyperedge_type1 + self.hyperedge_attr2 * (1 - hyperedge_type1) #超边的特征矩阵
        out = x1
        for ll in range(self.num_L):
            out = getattr(self, 'hyperconv%d' % (ll + 1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)
            # out = getattr(self, 'hyperconv%d' % (ll + 1))(out, hyperedge_index, self.hyperedge_weight[ll][0:hyperedge_index[1].max().item() + 1].to(edge_attr.device), edge_attr,self.EW_weight[ll][0:hyperedge_index.size(1)].to(edge_attr.device), dia_len)
            # out = getattr(self, 'hyperconv%d' % (ll + 1))(out, hyperedge_index, weight, edge_attr)  #新版torch_gec

        # if self.use_residue:
        # out1 = torch.cat([features, out], dim=-1)
            # out1 = self.reverse_features(dia_len, out1)

        # ---------------------------------------
        # gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        # gnn_out = x1
        # for kk in range(self.num_K):
        #     gnn_out = gnn_out + getattr(self, 'conv%d' % (kk + 1))(gnn_out, gnn_edge_index)
        #
        # weighted_HGNN = out * self.beita
        # weighted_GNN = gnn_out * (1-self.beita)
        # out2 = torch.cat([ weighted_HGNN,weighted_GNN], dim=1)
        # out2 = torch.cat([out, gnn_out], dim=1)
        out2 = out #用来删除多频
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        # ---------------------------------------
        return out1

    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i + 3):
                if _ < 3:
                    index2 = index2 + [edge_count] * i
                else:
                    index2 = index2 + [edge_count] * 3
                edge_count = edge_count + 1 #超边个数：3+N
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)  #三种模态的节点特征结合成一个矩阵
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i

            Gnodes = []
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_, 2))
                tmp = tmp + perm
            batch = batch + [batch_count] * i * 3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1] * i + [0] * 3

            node_count = node_count + i * num_modality

        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0, max_node + 1, 1).repeat_interleave(2).view(1, -1),
                               torch.arange(max_edge + 1, max_edge + 1 + max_node + 1, 1).repeat_interleave(2).view(1,
                                                                                                                    -1)],
                              dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1
            #hyperedge_index, (2,节点数x2) (点,超边) 有点类似与点与超边的关联矩阵  第一行是节点，第二行是超边 [nodeID, hyperedgeID]
            # edge_index,  (2,普通边数)
            # features,  (节点个数,节点特征)
            # batch,    (节点个数,)
            # hyperedge_type1 超边类型,分两种,0同模态超边,1跨模态超边   len为超边个数:3*batch_size+N (只有用到残差时才有用)

    def reverse_features(self, dia_len, features):
        l = []
        a = []
        v = []
        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index = []
        tmp = []

        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index = index + list(permutations(nodes_l, 2)) + list(permutations(nodes_a, 2)) + list(
                permutations(nodes_v, 2))
            Gnodes = []
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp + list(permutations(_, 2))
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i
            node_count = node_count + i * num_modality
        edge_index = torch.cat([torch.LongTensor(index).T, torch.LongTensor(tmp).T], 1).cuda()

        return edge_index, features
        #edge_index, (2,边数)  边数=[(句子数-1)*句子数]x3+句子数x3x2
        # features (节点数,节点特征)



    def create_hyper_index2(self, a, v, l, dia_len, modals, start,n_neighbors=2):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []

        # all_data = torch.cat([l, a, v], dim=1).cpu().detach().numpy()
        l_data = l.cpu().detach().numpy()
        a_data = a.cpu().detach().numpy()
        v_data = v.cpu().detach().numpy()
        nodes_count = 0
        index_2 = []
        index_1 = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]


            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto',metric='euclidean').fit(l_data[nodes_count:nodes_count + len(nodes_l)])
            _, indices = nbrs.kneighbors(l_data[nodes_count:nodes_count + len(nodes_l)])
            for j in indices:
                for k in j:
                    index_1.append(nodes_l[k])
                index_2.extend([n_neighbors * [edge_count + start]])
                edge_count += 1

            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto',metric='euclidean').fit(a_data[nodes_count:nodes_count + len(nodes_a)])
            _, indices = nbrs.kneighbors(a_data[nodes_count:nodes_count + len(nodes_a)])
            for j in indices:
                for k in j:
                    index_1.append(nodes_a[k])
                index_2.extend([n_neighbors * [edge_count + start]])
                edge_count += 1

            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto',metric='euclidean').fit(v_data[nodes_count:nodes_count + len(nodes_v)])
            _, indices = nbrs.kneighbors(v_data[nodes_count:nodes_count + len(nodes_v)])
            for j in indices:
                for k in j:
                    index_1.append(nodes_v[k])
                index_2.extend([n_neighbors * [edge_count + start]])
                edge_count += 1

            node_count = node_count + i * num_modality
        index_3 = []
        for i in index_2:
            index_3.extend(i)
        index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
        index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
        hyperedge_index = torch.cat([index_1, index_3], dim=0)
        # hyperedge_index_a = self.Clustering(a, nodes_a_all, start + len(nodes_l_all), n_clusters)
        # hyperedge_index_v = self.Clustering(v, nodes_v_all, start + len(nodes_l_all) + len(nodes_a_all), n_clusters)

        # hyperedge_index = torch.cat([hyperedge_index_l, hyperedge_index_a, hyperedge_index_v], dim=1)

        return hyperedge_index

    def Clustering(self,x_features, nodes__all, start, n_clusters):
        index1 = []
        index2 = []
        edge_count = 0
        count = 0
        x_features = x_features.cpu().detach().numpy()
        for nodes_index in nodes__all:
            all_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_features[count:count + len(nodes_index)])
            all_labels = all_kmeans.labels_
            all_unique_labels = set(all_labels)

            for label in all_unique_labels:
                indices = np.where(all_labels == label)[0]

                for idx in indices:
                    index1.append(idx + nodes_index[0])
                edge_count += 1
                index2.extend([len(indices) * [edge_count + start]])
            count += len(nodes_index)

        index3 = []
        for i in index2:
            index3.extend(i)
        index1 = torch.LongTensor(index1).view(1, -1).cuda()
        index3 = torch.LongTensor(index3).view(1, -1).cuda()
        hyperedge_index_single = torch.cat([index1, index3], dim=0)
        return hyperedge_index_single


    def create_hyper_index3(self, a, v, l, dia_len, modals, start, n_clusters=6):
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index1 = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            nodes_l_all.append(nodes_l)
            nodes_a_all.append(nodes_a)
            nodes_v_all.append(nodes_v)
            node_count += i * num_modality

        hyperedge_index_l = self.Clustering(l, nodes_l_all, start, n_clusters)
        hyperedge_index_a = self.Clustering(a, nodes_a_all, start + len(nodes_l_all), n_clusters)
        hyperedge_index_v = self.Clustering(v, nodes_v_all, start + len(nodes_l_all) + len(nodes_a_all), n_clusters)

        hyperedge_index = torch.cat([hyperedge_index_l, hyperedge_index_a, hyperedge_index_v], dim=1)

        return hyperedge_index

    def create_hyper_index5(self, a, v, l, dia_len, modals, start, n_clusters=6):
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index1 = []
        all_data = torch.cat([l, a, v], dim=1).cpu().detach().numpy()
        all_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_data)

        all_labels = all_kmeans.labels_
        all_unique_labels = set(all_labels)
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            nodes_l_all.append(nodes_l)
            nodes_a_all.append(nodes_a)
            nodes_v_all.append(nodes_v)
            node_count += i * num_modality

        hyperedge_index_l = self.Clustering(l, nodes_l_all, start, n_clusters)
        hyperedge_index_a = self.Clustering(a, nodes_a_all, start + len(nodes_l_all), n_clusters)
        hyperedge_index_v = self.Clustering(v, nodes_v_all, start + len(nodes_l_all) + len(nodes_a_all), n_clusters)

        hyperedge_index = torch.cat([hyperedge_index_l, hyperedge_index_a, hyperedge_index_v], dim=1)

        return hyperedge_index

    def create_hyper_index4(self, a, v, l, dia_len, modals, start, spk_idx):
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index0 = []
        index1 = []
        spk_idx = spk_idx.cpu().detach().numpy()
        index_1 = []
        index_2 = []
        edge_count=0
        nodes_count_l=0
        nodes_count_a = 0
        nodes_count_v = 0
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            # 使用enumerate模拟
            for idx,j in zip(nodes_l,spk_idx[nodes_count_l:nodes_count_l+len(nodes_l)]):
                # 使用idx和j做你需要的操作
                if j == 0:
                    index0.append(idx)
                elif j == 1:
                    index1.append(idx)
            index_2.extend([len(index0) * [edge_count + start]])
            index_2.extend([len(index1) * [edge_count + start +1]])
            index_1.extend(index0)
            index_1.extend(index1)
            # edge_count+=2
            # node_count += i * num_modality
            index0 = []
            index1 = []
            nodes_count_l += len(nodes_l)

            for idx,j in zip(nodes_a,spk_idx[nodes_count_a:nodes_count_a+len(nodes_a)]):
                # 使用idx和j做你需要的操作
                if j == 0:
                    index0.append(idx)
                elif j == 1:
                    index1.append(idx)
            index_2.extend([len(index0) * [edge_count + start+2]])
            index_2.extend([len(index1) * [edge_count + start +3]])
            index_1.extend(index0)
            index_1.extend(index1)
            # edge_count+=2
            # node_count += i * num_modality
            index0 = []
            index1 = []
            nodes_count_a += len(nodes_a)

            for idx,j in zip(nodes_v,spk_idx[nodes_count_v:nodes_count_v+len(nodes_v)]):
                # 使用idx和j做你需要的操作
                if j == 0:
                    index0.append(idx)
                elif j == 1:
                    index1.append(idx)
            index_2.extend([len(index0) * [edge_count + start+4]])
            index_2.extend([len(index1) * [edge_count + start +5]])
            index_1.extend(index0)
            index_1.extend(index1)
            edge_count+=6
            node_count += i * num_modality
            index0 = []
            index1 = []
            nodes_count_v += len(nodes_v)



        index_3=[]
        for i in index_2:
            index_3.extend(i)
        index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
        index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
        hyperedge_index = torch.cat([index_1, index_3], dim=0)
        # hyperedge_index_a = self.Clustering(a, nodes_a_all, start + len(nodes_l_all), n_clusters)
        # hyperedge_index_v = self.Clustering(v, nodes_v_all, start + len(nodes_l_all) + len(nodes_a_all), n_clusters)

        # hyperedge_index = torch.cat([hyperedge_index_l, hyperedge_index_a, hyperedge_index_v], dim=1)

        return hyperedge_index

    def create_hyper_index6(self, a, v, l, dia_len, modals, start, spk_idx,windows,step):
        """
        单个说话者局部超边
        """
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index0 = []
        index1 = []
        spk_idx = spk_idx.cpu().detach().numpy()
        index_1 = []
        index_2 = []
        edge_count=0
        nodes_count_l=0
        nodes_count_a = 0
        nodes_count_v = 0
        # windows = 5
        spk_tmp = 0
        # step = 6
        mode = "local"
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            # index1.extend(jvbu2(nodes_l,windows,step))
            # index1.extend(jvbu2(nodes_a,windows,step))
            # index1.extend(jvbu2(nodes_v,windows,step))

            index1.extend(single_people(nodes_l,windows,spk_idx[spk_tmp:spk_tmp+i],step,mode))
            index1.extend(single_people(nodes_a, windows, spk_idx[spk_tmp:spk_tmp+i],step,mode))
            index1.extend(single_people(nodes_v, windows, spk_idx[spk_tmp:spk_tmp+i],step,mode))
            spk_tmp+=i

            node_count += i * num_modality

        for i in index1:
            index_2.extend([len(i) * [edge_count + start]])
            edge_count+=1
        index_3 = []
        for i in index_2:
            index_3.extend(i)
        for i in index1:
            index_1.extend(i)
        index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
        index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
        hyperedge_index = torch.cat([index_1, index_3], dim=0)
        return hyperedge_index


    def create_hyper_index7(self, a, v, l, dia_len, modals, start, spk_idx):
        """
        单个说话者远距离超边fixed版
        """
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index0 = []
        index1 = []
        spk_idx = spk_idx.cpu().detach().numpy()
        index_1 = []
        index_2 = []
        edge_count = 0
        nodes_count_l = 0
        nodes_count_a = 0
        nodes_count_v = 0
        # windows = 5
        spk_tmp = 0
        # step = 6
        mode = "remote"
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            index1.extend(single_people(nodes_l, None, spk_idx[spk_tmp:spk_tmp + i + 1], None, mode))
            index1.extend(single_people(nodes_a, None, spk_idx[spk_tmp:spk_tmp + i + 1], None, mode))
            index1.extend(single_people(nodes_v, None, spk_idx[spk_tmp:spk_tmp + i + 1], None,mode))
            spk_tmp += i

            node_count += i * num_modality

        for i in index1:
            index_2.extend([len(i) * [edge_count + start]])
            edge_count += 1
        index_3 = []
        for i in index_2:
            index_3.extend(i)
        for i in index1:
            index_1.extend(i)
        index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
        index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
        hyperedge_index = torch.cat([index_1, index_3], dim=0)
        return hyperedge_index

    def create_hyper_index8(self, a, v, l, dia_len, modals, start, spk_idx,windows,step):
        """
        全部说话者局部超边
        """
        self_loop = False
        num_modality = len(modals)

        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()

        nodes_l_all = []
        nodes_a_all = []
        nodes_v_all = []
        num_modality = 3
        node_count = 0
        index0 = []
        index1 = []
        spk_idx = spk_idx.cpu().detach().numpy()
        index_1 = []
        index_2 = []
        edge_count=0
        nodes_count_l=0
        nodes_count_a = 0
        nodes_count_v = 0
        # windows = 5
        spk_tmp = 0
        # step = 6
        mode = "local"
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            index1.extend(jvbu2(nodes_l,windows,step))
            index1.extend(jvbu2(nodes_a,windows,step))
            index1.extend(jvbu2(nodes_v,windows,step))

            # index1.extend(single_people(nodes_l,windows,spk_idx[spk_tmp:spk_tmp+i+1],step,mode))
            # index1.extend(single_people(nodes_a, windows, spk_idx[spk_tmp:spk_tmp+i+1],step,mode))
            # index1.extend(single_people(nodes_v, windows, spk_idx[spk_tmp:spk_tmp+i+1],step,mode))
            # spk_tmp+=i

            node_count += i * num_modality

        for i in index1:
            index_2.extend([len(i) * [edge_count + start]])
            edge_count+=1
        index_3 = []
        for i in index_2:
            index_3.extend(i)
        for i in index1:
            index_1.extend(i)
        index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
        index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
        hyperedge_index = torch.cat([index_1, index_3], dim=0)
        return hyperedge_index