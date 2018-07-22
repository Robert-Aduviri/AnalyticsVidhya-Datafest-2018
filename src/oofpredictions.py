import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .neuralnet import train_model, eval_model, StructuredDataset, StructuredNet
from .utils import print_results

def test_catboost(model, X, y, X_test, kfolds, cat_features):
    trn_aucs, val_aucs = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    for trn_idx, val_idx in kfolds.split(X, y):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)],
                  use_best_model=True, cat_features=cat_features, 
                  verbose=False)
        y_trn_pred = model.predict_proba(X_trn)[:,1]
        y_val_pred = model.predict_proba(X_val)[:,1]
        trn_aucs.append(roc_auc_score(y_trn, y_trn_pred))
        val_aucs.append(roc_auc_score(y_val, y_val_pred))
        print(f'No. estimators: {model.tree_count_} | '
              f'Train AUC: {100*trn_aucs[-1]:.2f} | '
              f'Val AUC: {100*val_aucs[-1]:.2f}')
        
        y_tst_pred = model.predict_proba(X_test)[:,1]
        y_test += y_tst_pred
        y_pred[val_idx] = y_val_pred
        
    print()
    print_results(trn_aucs, val_aucs)
    print()
    return y_test / kfolds.n_splits, y_pred

def test_xgb(xgb, X, y, X_test, kfolds):
    trn_aucs, val_aucs = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    dtst = xgb.DMatrix(X_test)
    for trn_idx, val_idx in kfolds.split(X, y):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        dtrn = xgb.DMatrix(X_trn, y_trn)
        dval = xgb.DMatrix(X_val, y_val)
        evallist = [(dtrn, 'train'), (dval, 'eval')]
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
                  'max_depth': 3}
        num_rounds = 1000
        bst = xgb.train(params, dtrn, num_rounds, evallist,
                        early_stopping_rounds=30, verbose_eval=False)
        y_trn_pred = bst.predict(dtrn, ntree_limit=bst.best_ntree_limit)
        y_val_pred = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
        trn_aucs.append(roc_auc_score(y_trn, y_trn_pred))
        val_aucs.append(roc_auc_score(y_val, y_val_pred))
        print(f'No. estimators: {bst.best_ntree_limit} | '
              f'Train AUC: {100*trn_aucs[-1]:.2f} | '
              f'Val AUC: {100*val_aucs[-1]:.2f}')
        
        y_tst_pred = bst.predict(dtst, ntree_limit=bst.best_ntree_limit)
        y_test += y_tst_pred
        y_pred[val_idx] = y_val_pred
        
    print()
    print_results(trn_aucs, val_aucs)
    print()
    return y_test / kfolds.n_splits, y_pred

def test_lgbm(lgbm, X, y, X_test, kfolds, cat_features):
    trn_aucs, val_aucs = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    for trn_idx, val_idx in kfolds.split(X, y):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        dtrn = lgbm.Dataset(X_trn, y_trn)
        dval = lgbm.Dataset(X_val, y_val)
        params = {'objective':'binary', 'metric': 'auc', 'max_depth': 5}
        num_rounds = 1000
        bst = lgbm.train(params, dtrn, num_rounds, [dval],
                  early_stopping_rounds=30, categorical_feature=cat_features,
                  verbose_eval=False)
        y_trn_pred = bst.predict(X_trn)
        y_val_pred = bst.predict(X_val)
        trn_aucs.append(roc_auc_score(y_trn, y_trn_pred))
        val_aucs.append(roc_auc_score(y_val, y_val_pred))
        print(f'No. estimators: {bst.best_iteration} | '
              f'Train AUC: {100*trn_aucs[-1]:.2f} | '
              f'Val AUC: {100*val_aucs[-1]:.2f}')
        
        y_tst_pred = bst.predict(X_test)
        y_test += y_tst_pred
        y_pred[val_idx] = y_val_pred
        
    print()
    print_results(trn_aucs, val_aucs)
    print()
    return y_test / kfolds.n_splits, y_pred

def test_sklearn(model, X, y, X_test, kfolds):
    trn_aucs, val_aucs = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    for trn_idx, val_idx in kfolds.split(X, y):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        model.fit(X_trn, y_trn)
        y_trn_pred = model.predict_proba(X_trn)[:,1]
        y_val_pred = model.predict_proba(X_val)[:,1]
        trn_aucs.append(roc_auc_score(y_trn, y_trn_pred))
        val_aucs.append(roc_auc_score(y_val, y_val_pred))
        print(f'Train AUC: {100*trn_aucs[-1]:.2f} | '
              f'Val AUC: {100*val_aucs[-1]:.2f}')
        
        y_tst_pred = model.predict_proba(X_test)[:,1]
        y_test += y_tst_pred
        y_pred[val_idx] = y_val_pred
        
    print()
    print_results(trn_aucs, val_aucs)
    print()
    return y_test / kfolds.n_splits, y_pred

def test_neuralnet(X, y, X_test, kfolds, cat_cols, num_cols, USE_CUDA=False):
    cat_szs = [int(X[col].max() + 1) for col in cat_cols]
    emb_szs = [(c, min(50, (c+1)//2)) for c in cat_szs]
    trn_aucs, val_aucs = [], []
    y_pred = np.zeros(len(X))
    y_test = np.zeros(len(X_test))
    test_dl = DataLoader(StructuredDataset(
                        X_test[cat_cols], X_test[num_cols]),
                        batch_size=32)
    for trn_idx, val_idx in kfolds.split(X, y):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        
        model = StructuredNet(emb_szs, n_cont=len(num_cols), emb_drop=0.2,
                      szs=[1000,500], drops=[0.5, 0.5])
        if USE_CUDA: model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        train_dl = DataLoader(StructuredDataset(
                        X_trn[cat_cols], X_trn[num_cols], y_trn),
                        batch_size=32, shuffle=True)
        val_dl = DataLoader(StructuredDataset(
                        X_val[cat_cols], X_val[num_cols], y_val),
                        batch_size=32)
    
        best_epoch = train_model(model, train_dl, val_dl, optimizer, criterion,
                    n_epochs=7, USE_CUDA=USE_CUDA)    
        model.load_state_dict(torch.load(f'data/neuralnet/model_e{best_epoch}.pt'))
        
        train_dl = DataLoader(StructuredDataset(
                        X_trn[cat_cols], X_trn[num_cols], y_trn),
                        batch_size=32)
        
        _, y_trn_pred = eval_model(model, train_dl, USE_CUDA=USE_CUDA)
        _, y_val_pred = eval_model(model, val_dl, USE_CUDA=USE_CUDA)
        trn_aucs.append(roc_auc_score(y_trn, y_trn_pred))
        val_aucs.append(roc_auc_score(y_val, y_val_pred))
        print(f'Best epoch: {best_epoch+1} | '
              f'Train AUC: {100*trn_aucs[-1]:.2f} | '
              f'Val AUC: {100*val_aucs[-1]:.2f}')
        print()
        
        _, y_tst_pred = eval_model(model, test_dl, USE_CUDA=USE_CUDA)
        y_test += y_tst_pred
        y_pred[val_idx] = y_val_pred
        
    print()
    print_results(trn_aucs, val_aucs)
    print()
    return y_test / kfolds.n_splits, y_pred