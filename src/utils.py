import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .neuralnet import train_model, eval_model, StructuredDataset, StructuredNet

#### PREPROCESSING ####

def apply_cats(df, trn):
    """Changes any columns of strings in df (DataFrame) into categorical variables
    using trn (DataFrame) as a template for the category codes (inplace)."""
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)

def to_cat_codes(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.as_ordered()
        
def label_encode(train, test, cat_cols):
    def get_dict(labels):
        return {label: idx for idx, label in enumerate(labels)}

    labels = {
        'enrolled_university': get_dict(['no_enrollment', 'Part time course',
                                         'Full time course']),
        'education_level': get_dict(['Primary School', 'High School', 
                                     'Graduate', 'Masters', 'Phd']),
        'experience': get_dict(['<1'] + \
                               [str(x) for x in range(1,21)] + ['>20']),
        'company_size': get_dict(['<10', '10/49', '50-99', '100-500', 
                                  '500-999', 
                                  '1000-4999', '5000-9999', '10000+']),
        'last_new_job': get_dict([str(x) for x in range(1,5)] + \
                                 ['>4', 'never'])
    }

    for col in labels:
        train[col] = train[col].map(labels[col])
        test[col] = test[col].map(labels[col])

    from src.utils import to_cat_codes, apply_cats
    to_cat_codes(train, [c for c in cat_cols if c not in labels])
    apply_cats(test, train)
    for col in cat_cols: 
        if col not in labels:
            train[col] = train[col].cat.codes
            test[col] = test[col].cat.codes

    return labels
            
def fill_na(train, test, cat_cols):
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    nan_cols = [c for c in cat_cols if \
                any(df[c].min() < 0 for df in [train, test])]

    for c in nan_cols:
        train[c] = train[c] + 1
        test[c] = test[c] + 1
        
def add_num_features(train, test, labels, num_cols):
    for col in labels:
        train[f'{col}_num'] = train[col]
        test[f'{col}_num'] = test[col]
        num_cols.append(f'{col}_num')
        
def scale_num_features(train, test, num_cols):
    scaler = StandardScaler().fit(pd.concat([train[num_cols], 
                                             test[num_cols]]))

    def scale_features(df, scaler, num_cols):
        scaled = scaler.transform(df[num_cols])
        for i, col in enumerate(num_cols):
            df[col] = scaled[:,i]

    scale_features(train, scaler, num_cols)
    scale_features(test, scaler, num_cols)
    
def preprocess(train, test, cat_cols, num_cols):
    labels = label_encode(train, test, cat_cols)
    fill_na(train, test, cat_cols)
    add_num_features(train, test, labels, num_cols)
    scale_num_features(train, test, num_cols)
    
#### MODEL #### 

def eval_catboost(model, X, y, kfolds, cat_features):
    trn_aucs, val_aucs = [], []
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
    print()
    return trn_aucs, val_aucs

def eval_xgb(xgb, X, y, kfolds):
    trn_aucs, val_aucs = [], []
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
    print()
    return trn_aucs, val_aucs
    
def eval_lgbm(lgbm, X, y, kfolds, cat_features):
    trn_aucs, val_aucs = [], []
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
    print()
    return trn_aucs, val_aucs

def eval_tree(model, X, y, kfolds):
    trn_aucs, val_aucs = [], []
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
    print()
    return trn_aucs, val_aucs
    
def eval_neuralnet(X, y, kfolds, cat_cols, num_cols, USE_CUDA=False):
    cat_szs = [int(X[col].max() + 1) for col in cat_cols]
    emb_szs = [(c, min(50, (c+1)//2)) for c in cat_szs]
    trn_aucs, val_aucs = [], []
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
    print()
    return trn_aucs, val_aucs
    
def print_results(trn_aucs, val_aucs):
    print(f'{100*np.mean(trn_aucs):.2f} +/- {200*np.std(trn_aucs):.2f} | '
          f'{100*np.mean(val_aucs):.2f} +/- {200*np.std(val_aucs):.2f}')
    

