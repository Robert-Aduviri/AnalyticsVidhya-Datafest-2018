import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset

class StructuredDataset(Dataset):
    def __init__(self, cats, conts, targets=None):
        self.cats = cats.values.astype(np.int64)
        self.conts = conts.values.astype(np.float32)
        self.targets = np.array(targets).astype(np.float32) \
                          if targets is not None else \
                          np.zeros(len(cats)).astype(np.float32)
    
    def __len__(self):
        return len(self.cats)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.targets[idx]]

class StructuredNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops):
        super().__init__()        
        self.embs = nn.ModuleList([
            nn.Embedding(c, s) for c,s in emb_szs
        ])
        for emb in self.embs:
            self.emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont
        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)
        ])
        for o in self.lins:
            nn.init.kaiming_normal_(o.weight.data)
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]
        ])
        self.outp = nn.Linear(szs[-1], 1)
        nn.init.kaiming_normal_(self.outp.weight.data)
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([
            nn.Dropout(drop) for drop in drops
        ])
        self.bn = nn.BatchNorm1d(n_cont)
        
    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn(x_cont)
        x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for lin, drop, bn in zip(self.lins, self.drops, self.bns):
            x = F.relu(lin(x))
            x = bn(x)
            x = drop(x)
        return self.outp(x)
    
    def emb_init(self, x):
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)
        
def train_step(model, cats, conts, targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    preds = model(cats, conts)
    loss = criterion(preds.view(-1), targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_model(model, data_loader, USE_CUDA=False):
    targets, preds = [], []
    model.eval()
    for batch_idx, (cats, conts, target) in enumerate(data_loader):
        with torch.no_grad():
            if USE_CUDA:
                cats, conts, target = cats.cuda(), conts.cuda(), target.cuda()
            pred = model(cats, conts)
            targets.extend(target.cpu())
            preds.extend(pred.cpu())
            assert len(targets) == len(preds)
    return [x.item() for x in targets], [F.sigmoid(x).item() for x in preds]

def get_metrics(model, data_loader, USE_CUDA=False):
    targets, preds = eval_model(model, data_loader, USE_CUDA=USE_CUDA)
    return roc_auc_score(targets, preds)

def train_model(model, train_loader, val_loader, optimizer, criterion,
                n_epochs, USE_CUDA=False):
    val_auc_scores = []
    for epoch in range(n_epochs):
        for batch_idx, (cats, conts, target) in enumerate(train_loader):
            if USE_CUDA:
                cats, conts, target = cats.cuda(), conts.cuda(), target.cuda()
            train_step(model, cats, conts, target, optimizer, criterion)
        
        if val_loader is not None:
            train_auc = get_metrics(model, train_loader, USE_CUDA)
            val_auc = get_metrics(model, val_loader, USE_CUDA)
            print(f'Epoch: {epoch+1} | Train AUC: {100*train_auc:.2f} | '
                  f'Val AUC: {100*val_auc:.2f}')
            val_auc_scores.append(val_auc)
            
        torch.save(model.state_dict(), f'data/neuralnet/model_e{epoch}.pt')    
        
    return max(range(len(val_auc_scores)), key=lambda x: val_auc_scores[x])