import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from gat import GAT
import numpy as np

class LMACL(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, temp, lambda_1, lambda_2,
                 num_hidden, num_heads, num_layers, in_drop, attn_drop, negative_slope,
                 Graph, device):
        super(LMACL, self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.n_u = n_u
        self.n_i = n_i
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.d = d
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l + 1)
        self.Z_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.act = nn.LeakyReLU(0.5)
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.Graph = Graph
        self.E_u = None
        self.E_i = None
        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test == True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1 - mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions

        else:  # training phase
            # multi-GAT propagation
            heads = ([self.num_heads] * self.num_layers)
            g = dgl.from_scipy(self.Graph)
            if torch.cuda.is_available():
                cuda = True
                g = g.int().to(self.device)
            else:
                cuda = False
            # add self loop
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)

            ma = GAT(g,
                     self.num_layers,
                     self.d,
                     self.num_hidden,
                     heads,
                     F.elu,
                     self.in_drop,
                     self.attn_drop,
                     self.negative_slope)

            for layer in range(1, self.l + 1):
                # GCN propagation
                self.Z_u_list[layer] = torch.spmm(self.adj_norm, self.E_i_list[layer - 1])
                self.Z_i_list[layer] = torch.spmm(self.adj_norm.transpose(0, 1), self.E_u_list[layer - 1])
                # GAT
                if cuda:
                    ma.cuda()
                    self.E_u_list[layer - 1] = self.E_u_list[layer - 1].cuda()
                    self.E_i_list[layer - 1] = self.E_i_list[layer - 1].cuda()
                ls = torch.cat((self.E_u_list[layer - 1], self.E_i_list[layer - 1]), dim=0)
                ls1 = ma(ls)
                self.G_u_list[layer], self.G_i_list[layer] = torch.split(ls1, [self.n_u, self.n_i], dim=0)
                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer - 1]
                self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer - 1]

            # aggregate across layers
            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = torch.log(torch.exp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp)).mean() + \
                        torch.log(torch.exp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp)).mean()
            loss_s = -pos_score + neg_score

            # gcn bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s