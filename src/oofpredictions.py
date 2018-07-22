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
    y_pred = np.zeros_like(y)
    y_test = np.zeros_like(y)
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
    return y_test / kfolds.splits(), y_pred