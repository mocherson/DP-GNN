import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

import os

from tqdm import tqdm

from util import *
from model import *

parser = argparse.ArgumentParser(description='PyTorch graph neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG",
                    help='name of dataset (default: MUTAG)')
parser.add_argument('--datapath', type=str, default="./data/",
                    help='path of dataset (default: ./data/)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--fold_idx', type=int, default=0,
                    help='the index of fold in 10-fold validation. Should be less then 10.')
parser.add_argument('--folds', type=int, default=10,
                    help='the number of folds in n-fold validation.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
parser.add_argument('--agg', type=str, default="cat", choices=["cat", "sum", "mean"],
                    help='aggregate input and its neighbors, can be extended to other method like mean, max etc.')
parser.add_argument('--attribute', action="store_true",
                    help='Whether it is for attributed graph.')
parser.add_argument('--phi', type=str, default="MLP", choices=["power", "identical", "MLP","vdmd"],
                    help='transformation before aggregation')
parser.add_argument('--first_phi', action="store_true",
                    help='Whether using phi for first layer. False indicates no transform')
parser.add_argument('--dropout', type=float, default=0,
                        help='final layer dropout (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay in the optimizer (default: 0)')
parser.add_argument('--filename', type = str, default = "",
                    help='save result to file')
parser.add_argument('--balance', action="store_true",
                    help='whether balance the labels by different weights')
parser.add_argument('--wt', type=str, default="10000",
                    help='the weight of different tasks')
args = parser.parse_args()

device = args.device
dataset = args.dataset
fold_idx = args.fold_idx+1 
agg = args.agg
hid_dim = args.hidden_dim
dropout = args.dropout
isattr = args.attribute
weight_decay = args.weight_decay
firstphi = args.first_phi
datapath = args.datapath
folds=args.folds

wt = args.wt
# wt = '10000'
if not args.filename == "":
    filename = args.filename  
else:
    filename = "./results/{}/{}folds/weight{}/{}_{}_hid{}_wd{}_{}.csv".format(dataset,folds, wt, args.phi,fold_idx,hid_dim, weight_decay, agg )
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    if not os.path.isdir("./results/{}".format(dataset)):
        os.mkdir("./results/{}".format(dataset))
    if not os.path.isdir("./results/{}/{}folds".format(dataset, folds)):
        os.mkdir("./results/{}/{}folds".format(dataset, folds)) 
    if not os.path.isdir("./results/{}/{}folds/weight{}".format(dataset,folds, wt)):
        os.mkdir("./results/{}/{}folds/weight{}".format(dataset,folds, wt)) 
    if not os.path.isdir("./results/{}/{}folds/embeddings".format(dataset,folds)):    
        os.mkdir("./results/{}/{}folds/embeddings".format(dataset,folds))
    if not os.path.isdir("./results/{}/{}folds/embeddings/weight{}".format(dataset,folds, wt)): 
        os.mkdir("./results/{}/{}folds/embeddings/weight{}".format(dataset,folds, wt))
if os.path.isfile(filename):
    print('%s, file exists.'%(filename))
    os._exit(0)

torch.manual_seed(0)
np.random.seed(0)    
device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

bgraph=[]    
def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    idxs = np.random.permutation(len(train_graphs))
    
    i=0
    loss_accum, loss0_accum,loss1_accum,loss2_accum,loss3_accum,loss4_accum = 0,0,0,0,0,0
    while i<len(idxs):
        selected_idx = idxs[i:i+args.batch_size]
        i = i+args.batch_size

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        loss, loss0,loss1,loss2,loss3,loss4 = model(batch_graph)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()
        loss0_accum += loss0.item()
        loss1_accum += loss1.item()
        loss2_accum += loss2.item()
        loss3_accum += loss3.item()
        loss4_accum += loss4.item()

    print("epoch:%d, total loss=%f, loss0=%f, loss1=%f, loss2=%f, loss3=%f, loss4=%f" % (epoch, loss_accum, \
             loss0_accum,loss1_accum,loss2_accum,loss3_accum,loss4_accum))
    
    return loss_accum, loss0_accum,loss1_accum,loss2_accum,loss3_accum,loss4_accum


def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    with torch.no_grad():
#         acc_train=0
#         if sum([len(g.node_tags) for g in train_graphs])<120000:
        correct, n_graphs, emb_tr, _ = model(train_graphs)
        acc_train = correct / n_graphs

        correct, n_graphs, emb_te, _ = model(test_graphs)
        acc_test = correct / n_graphs
        
        if epoch%10==0 and args.fold_idx==0:
            save_obj(emb_tr.cpu().numpy(), './results/{}/{}folds/embeddings/weight{}/tr_{}_idx{}_hid{}_{}_ep{}.pkl'.format(dataset, folds, wt, args.phi, fold_idx,hid_dim,agg,epoch))
            save_obj(emb_te.cpu().numpy(), './results/{}/{}folds/embeddings/weight{}/te_{}_idx{}_hid{}_{}_ep{}.pkl'.format(dataset, folds, wt, args.phi, fold_idx, hid_dim,agg,epoch))

    print("accuracy train: %f,  test: %f" % (acc_train,  acc_test))

    return acc_train, acc_test

if isattr:
    graphs, num_classes = load_data_general(dataset,datapath)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
else:
    train_graphs, test_graphs, val_graphs, num_classes, num_node_classes = load_train_test(dataset, fold_idx, nfolds=args.folds, path=datapath)


m = max([graph.max_neighbor for graph in train_graphs])
in_dim = train_graphs[0].node_features.shape[1]
out_features = ((hid_dim, hid_dim ), (hid_dim, hid_dim ), (hid_dim, hid_dim ), (hid_dim, hid_dim), (hid_dim, hid_dim))

if args.phi=="power":
    if firstphi:
        phi_features = (in_dim*m+1, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [PHI(m) for i in range(5)]
    else:
        phi_features = (in_dim, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [lambda x:x]+[PHI(m) for i in range(4)]
elif args.phi=="identical":
    phi_features = (in_dim, hid_dim,hid_dim,hid_dim,hid_dim)
    ph = [lambda x:x]*5
elif args.phi=="MLP":
    phi_features = (in_dim, hid_dim,hid_dim,hid_dim,hid_dim)
    if firstphi:
        ph = [MLP(in_dim,(hid_dim,in_dim), batch_norm=True)]+[MLP(hid_dim,(hid_dim,hid_dim), batch_norm=True) for i in range(4)]
    else:
        ph = [lambda x:x]+[MLP(hid_dim,(hid_dim,hid_dim), batch_norm=True) for i in range(4)]
elif args.phi == "vdmd":
    if firstphi:
        phi_features = (in_dim*m+1, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [vdPHI(m) for i in range(5)]
    else:
        phi_features = (in_dim, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [lambda x:x]+[vdPHI(m) for i in range(4)]

if args.balance:
    g_cnt = Counter([graph.label for graph in graphs])
    graph_weight = [sum(g_cnt.values())/g_cnt[i] for i in range(len(g_cnt))]
    n_cnt = Counter(sum([graph.node_tags for graph in graphs],[]))
    node_weight = [sum(n_cnt.values())/n_cnt[i] for i in range(len(n_cnt))]
    n_cnt_g = Counter(sum([[graph.label]*len(graph.node_tags) for graph in graphs], []))
    node_weight_g=[sum(n_cnt_g.values())/n_cnt_g[i] for i in range(len(n_cnt_g))]
else:
    graph_weight=None
    node_weight=None
    node_weight_g=None

model = ExpGraphNN_MTL(in_dim,phi_features,out_features, n_graph_class=num_classes, n_node_class=num_node_classes, 
                       graph_weight=graph_weight, node_weight=node_weight, node_weight_g=node_weight_g, 
                       dropout=dropout, phis=ph, batch_norm=True, mtl_weight=[int(s) for s in wt], agg=agg).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

acc_tr=[]
acc_te=[]
loss, loss0,loss1,loss2,loss3,loss4 =[],[],[],[],[],[],
bestacc=0
bestloss=np.inf
best_epoc = 0
for epoch in range(1, args.epochs + 1):
    scheduler.step()
    loss_accum, loss0_accum,loss1_accum,loss2_accum,loss3_accum,loss4_accum = train(args, model, device, train_graphs, optimizer, epoch)
    acc_train, acc_test = test(args, model, device, train_graphs, test_graphs,  epoch)
   
    acc_tr.append(acc_train)
    acc_te.append(acc_test)
    loss.append(loss_accum)
    loss0.append(loss0_accum)
    loss1.append(loss1_accum)
    loss2.append(loss2_accum)
    loss3.append(loss3_accum)
    loss4.append(loss4_accum)
    
#     if acc_train>bestacc or avg_loss<bestloss:
#         bestacc=max(acc_train, bestacc)
#         bestloss=min(avg_loss, bestloss)
#         best_epoc=epoch
        
#     if epoch-best_epoc>=50:
#         break

res = pd.DataFrame({"acc_tr":acc_tr,"acc_te":acc_te,"loss":loss, "loss0":loss0,"loss1":loss1,"loss2":loss2,"loss3":loss3,"loss4":loss4})    

res.to_csv(filename)

    


