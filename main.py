import numpy as np
import torch
import pickle
from model import LMACL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parsers import args
from tqdm import tqdm
import os
from scipy import sparse
import torch.utils.data as data
from utils import TrnData
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lr = args.lr
num_hidden = args.num_hidden
num_heads = args.num_heads
num_layers = args.num_layers
in_drop = args.in_drop
attn_drop = args.attn_drop
negative_slope = args.negative_slope

# load data
path = 'data/' + args.data + '/'
f = open(path + 'train.pkl', 'rb')
train = pickle.load(f)
train_csr = (train != 0).astype(np.float32)
f = open(path + 'test.pkl', 'rb')
test = pickle.load(f)

test_np = test.toarray()
print('Data loaded.')

print('user_num:', train.shape[0], 'item_num:', train.shape[1], 'lr:', lr, 'num_heads:', num_heads, 'temp:', temp, 'l:',
      l, 'epoch:', epoch_no)

epoch_user = min(train.shape[0], 30000)

# A
adj_mat = sparse.dok_matrix((train.shape[0] + train.shape[1], train.shape[0] + train.shape[1]), dtype=np.float32)
adj_mat = adj_mat.tolil()
R = sparse.csr_matrix(train).tolil()
adj_mat[:train.shape[0], train.shape[0]:] = R
adj_mat[train.shape[0]:, :train.shape[0]] = R.T
Graph = adj_mat.todok()

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LMACL(adj_norm.shape[0], adj_norm.shape[1], d, train_csr, adj_norm, l, temp, lambda_1, lambda_2,
                 num_hidden, num_heads, num_layers, in_drop, attn_drop, negative_slope, Graph, device)
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)


def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] * 0.96
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr


current_lr = lr
recall20Max = 0
ndcg20Max = 0
recall40Max = 0
ndcg40Max = 0
bestEpoch = 0

for epoch in range(epoch_no):
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'saved_model/saved_model_epoch_' + str(epoch) + '.pt')
        torch.save(optimizer.state_dict(), 'saved_model/saved_optim_epoch_' + str(epoch) + '.pt')

    current_lr = learning_rate_decay(optimizer)
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()

    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)
        iids = iids.long().cuda(torch.device(device))
        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s = model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        # print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss / batch_no
    epoch_loss_r = epoch_loss_r / batch_no
    epoch_loss_s = epoch_loss_s / batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 1 == 0:  # test every 1 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids) / batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in tqdm(range(batch_no)):
            start = batch * batch_user
            end = min((batch + 1) * batch_user, len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input, None, None, None, test=True)
            predictions = np.array(predictions.cpu())

            # top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end], predictions, 20, test_labels)
            # top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end], predictions, 40, test_labels)

            all_recall_20 += recall_20
            all_ndcg_20 += ndcg_20
            all_recall_40 += recall_40
            all_ndcg_40 += ndcg_40
        print('-------------------------------------------')
        print('Test of epoch', epoch, ':', 'Recall@20:', all_recall_20 / batch_no, 'Ndcg@20:', all_ndcg_20 / batch_no,
              'Recall@40:', all_recall_40 / batch_no, 'Ndcg@40:', all_ndcg_40 / batch_no)

        if all_recall_20 / batch_no > recall20Max:
            recall20Max = all_recall_20 / batch_no
            ndcg20Max = all_ndcg_20 / batch_no
            recall40Max = all_recall_40 / batch_no
            ndcg40Max = all_ndcg_40 / batch_no
            bestEpoch = epoch

        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20 / batch_no)
        ndcg_20_y.append(all_ndcg_20 / batch_no)
        recall_40_y.append(all_recall_40 / batch_no)
        ndcg_40_y.append(all_ndcg_40 / batch_no)

print('Best epoch : ', bestEpoch, ' , Recall20 : ', recall20Max, ' , NDCG20 : ', ndcg20Max, ' , Recall40 : ',
      recall40Max, ' , NDCG40 : ', ndcg40Max, )

metric = pd.DataFrame({
    'epoch': recall_20_x,
    'recall@20': recall_20_y,
    'ndcg@20': ndcg_20_y,
    'recall@40': recall_40_y,
    'ndcg@40': ndcg_40_y
})
current_t = time.gmtime()
metric.to_csv('log/result_' + args.data + '_' + time.strftime('%Y-%m-%d-%H', current_t) + '.csv')

torch.save(model.state_dict(),
           'saved_model/saved_model_' + args.data + '_' + time.strftime('%Y-%m-%d-%H', current_t) + '.pt')
torch.save(optimizer.state_dict(),
           'saved_model/saved_optim_' + args.data + '_' + time.strftime('%Y-%m-%d-%H', current_t) + '.pt')

