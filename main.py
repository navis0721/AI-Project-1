import argparse
import pandas as pd
import numpy as np
import torch
import random
from datetime import datetime
import logging
import time
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import copy
from model import LSTM_model

def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--static_size', type=int, default=5)
    parser.add_argument('--feature_size', type=int, default=20)
    parser.add_argument('--static_pca_size', type=int, default=1)
    parser.add_argument('--feature_pca_size', type=int, default=4)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--treatment_dim', type=int, default=1)
    parser.add_argument('--treatment_emb_dim', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_smote', type=bool, default=False)
    parser.add_argument('--use_pca', type=bool, default=False)
    parser.add_argument('--log_file', type=str, default='log/{}/{}.log'.format(date, hour))
    parser.add_argument('--dataset_path', type=str, default="datasets/syn_dataset_hidden0_noise0_5k.pt")
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


timestamp = datetime.now().strftime('%Y-%m%d_%H%M%S')
date, hour = timestamp.split('_')

def get_loss_value(y_true, y_pred, loss_type="mse"):
    loss_type = loss_type.lower()

    # torch version
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        if loss_type == "mse":
            return torch.mean((y_true - y_pred) ** 2)
        elif loss_type == "mae":
            return torch.mean(torch.abs(y_true - y_pred))
        else:
            raise ValueError("loss_type must be 'mse' or 'mae'")

    # numpy / sklearn version
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if loss_type == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif loss_type == "mae":
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError("loss_type must be 'mse' or 'mae'")

def subset_to_tensors(subset):
    """
    subset: Subset(TensorDataset(...))
    returns:
        X_static [N, D_static]
        X_time   [N, T, D_time]
        X_treat  [N, T, D_treat]
        y        [N, T, ...]
    """
    X_static_list, X_time_list, X_treat_list, y_list = [], [], [], []

    for X_static, X_time, X_treat, y in subset:
        if X_static.ndim == 1:
            X_static = X_static.unsqueeze(0)
            X_time = X_time.unsqueeze(0)
            X_treat = X_treat.unsqueeze(0)
            y = y.unsqueeze(0)

        X_static_list.append(X_static)
        X_time_list.append(X_time)
        X_treat_list.append(X_treat)
        y_list.append(y)

    X_static = torch.cat(X_static_list, dim=0)
    X_time = torch.cat(X_time_list, dim=0)
    X_treat = torch.cat(X_treat_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X_static, X_time, X_treat, y

def apply_smote_to_train_subset(train_subset, treatment_time_index=0, random_state=66):
    """
    只對 train subset 做 SMOTE，使用:
      - statics
      - feature
    來平衡 treated/control

    treatment label 取自某個固定時間點 treatment_time_index

    回傳 balanced train TensorDataset
    """
    X_static, X_time, X_treat, y = subset_to_tensors(train_subset)

    # ===== 取 treated/control label =====
    if X_treat.ndim == 3:
        # [N, T, 1] or [N, T, D_treat]
        label = X_treat[:, treatment_time_index, 0]
    elif X_treat.ndim == 2:
        # [N, T]
        label = X_treat[:, treatment_time_index]
    else:
        raise ValueError(f"Unexpected X_treat shape: {X_treat.shape}")

    label = label.long().cpu().numpy()

    unique_labels = np.unique(label)
    if len(unique_labels) != 2:
        raise ValueError(f"SMOTE requires binary classes, but got labels: {unique_labels}")

    # ===== flatten statics + time-varying features for SMOTE =====
    N, T, D_time = X_time.shape
    D_static = X_static.shape[1]

    X_static_np = X_static.cpu().numpy()
    X_time_np = X_time.cpu().numpy().reshape(N, -1)

    X_for_smote = np.concatenate([X_static_np, X_time_np], axis=1)

    # ===== 讓 y 和 X_treat 也跟著重建 =====
    y_np = y.cpu().numpy().reshape(N, -1)
    X_treat_np = X_treat.cpu().numpy().reshape(N, -1)

    pack_all = np.concatenate([X_for_smote, X_treat_np, y_np], axis=1)

    smote = SMOTE(random_state=random_state)
    pack_resampled, label_resampled = smote.fit_resample(pack_all, label)

    # ===== split back =====
    x_end = D_static + T * D_time
    treat_end = x_end + X_treat_np.shape[1]

    X_res = pack_resampled[:, :x_end]
    X_treat_res = pack_resampled[:, x_end:treat_end]
    y_res = pack_resampled[:, treat_end:]

    X_static_res = X_res[:, :D_static]
    X_time_res = X_res[:, D_static:].reshape(-1, T, D_time)

    X_treat_res = X_treat_res.reshape(-1, *X_treat.shape[1:])
    y_res = y_res.reshape(-1, *y.shape[1:])

    # treatment 用二元化，避免 SMOTE 插值後出現 0~1 之間的小數
    X_treat_res = (X_treat_res >= 0.5).astype(np.float32)

    balanced_train_dataset = TensorDataset(
        torch.tensor(X_static_res, dtype=torch.float32),
        torch.tensor(X_time_res, dtype=torch.float32),
        torch.tensor(X_treat_res, dtype=torch.float32),
        torch.tensor(y_res, dtype=torch.float32),
    )

    return balanced_train_dataset

def fit_pca_on_train(
    train_subset,
    n_components_static=None,
    n_components_time=None,
):
    """
    只用 train subset fit PCA
    """
    X_static, X_time, _, _ = subset_to_tensors(train_subset)

    # static: [N, D_static]
    X_static_np = X_static.cpu().numpy()

    # time: [N, T, D_time] -> [N*T, D_time]
    N, T, D_time = X_time.shape
    X_time_np = X_time.cpu().numpy().reshape(N * T, D_time)

    pca_static = None
    pca_time = None

    if n_components_static is not None and n_components_static < X_static_np.shape[1]:
        pca_static = PCA(n_components=n_components_static)
        pca_static.fit(X_static_np)

    if n_components_time is not None and n_components_time < D_time:
        pca_time = PCA(n_components=n_components_time)
        pca_time.fit(X_time_np)

    return pca_static, pca_time

def apply_pca_to_fold(
    train_subset,
    val_subset,
    n_components_static=None,
    n_components_time=None,
):
    pca_static, pca_time = fit_pca_on_train(
        train_subset=train_subset,
        n_components_static=n_components_static,
        n_components_time=n_components_time,
    )

    train_dataset_pca = transform_subset_with_pca(
        train_subset,
        pca_static=pca_static,
        pca_time=pca_time,
    )

    val_dataset_pca = transform_subset_with_pca(
        val_subset,
        pca_static=pca_static,
        pca_time=pca_time,
    )

    return train_dataset_pca, val_dataset_pca

def transform_subset_with_pca(subset, pca_static=None, pca_time=None):
    """
    將 subset 轉成 PCA 後的新 TensorDataset
    """
    X_static, X_time, X_treat, y = subset_to_tensors(subset)

    X_static_np = X_static.cpu().numpy()
    X_time_np = X_time.cpu().numpy()
    X_treat_np = X_treat.cpu().numpy()

    N, T, D_time = X_time_np.shape

    # static
    if pca_static is not None:
        X_static_new = pca_static.transform(X_static_np)
    else:
        X_static_new = X_static_np

    # time
    if pca_time is not None:
        X_time_flat = X_time_np.reshape(N * T, D_time)
        X_time_new = pca_time.transform(X_time_flat).reshape(N, T, -1)
    else:
        X_time_new = X_time_np
    
    # treatment不做pca
    X_treat_new = X_treat_np

    new_dataset = TensorDataset(
        torch.tensor(X_static_new, dtype=torch.float32),
        torch.tensor(X_time_new, dtype=torch.float32),
        torch.tensor(X_treat_new, dtype=torch.float32),
        y.float(),
    )
    return new_dataset

def build_lr_input(X_static, X_time, X_treat):
    """
    X_static: [B, D_static]
    X_time  : [B, T, D_time]
    X_treat : [B, T, D_treat]

    return:
        X_lr: [B*T, D_static + D_time + D_treat]
    """
    X_static = X_static.unsqueeze(1).repeat(1, X_time.shape[1], 1)
    X = torch.cat((X_static, X_time, X_treat), dim=-1)
    X = X.reshape(-1, X.shape[-1])
    return X

def train_eval_lstm(train_loader, val_loader, args, device, LSTM_model):
    model = LSTM_model(args=args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-8)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for X_static, X_time, X_treat, y in train_loader:
            optimizer.zero_grad()

            X_static = X_static.to(device)
            X_time = X_time.to(device)
            X_treat = X_treat.to(device)
            y = y[:, :, 0].to(device)   # factual outcome

            if args.model == 'LSTM':
                preds = model(X_static, X_time, X_treat)

            if args.loss == "mse" or args.loss == "pehe":
                loss = get_loss_value(preds.reshape(-1), y.reshape(-1), loss_type="mse")
            elif args.loss == "mae" or args.loss == "eate":
                loss = get_loss_value(preds.reshape(-1), y.reshape(-1), loss_type="mae")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_static, X_time, X_treat, y in val_loader:
                X_static = X_static.to(device)
                X_time = X_time.to(device)
                X_treat = X_treat.to(device)
                y_f = y[:, :, 0].to(device)
                y_cf = y[:, :, 1].to(device)

                preds = model(X_static, X_time, X_treat)
                preds_cf = model(X_static, X_time, 1 - X_treat)
                if args.loss == "mse" or args.loss == "mae":
                    loss = get_loss_value(preds.reshape(-1), y_f.reshape(-1), loss_type=args.loss)
                elif args.loss == "pehe" or args.loss == "eate":
                    pred_ite = preds - preds_cf
                    true_ite = y_f - y_cf
                    if args.loss == "pehe":
                        loss = get_loss_value(pred_ite.reshape(-1), true_ite.reshape(-1), loss_type="mse")
                    elif args.loss == "eate":
                        loss = get_loss_value(pred_ite.reshape(-1), true_ite.reshape(-1), loss_type="mae")
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

def collect_from_subset(subset):
    X_static_all = []
    X_time_all = []
    X_treat_all = []
    y_all = []

    for X_static, X_time, X_treat, y in subset:
        if X_static.ndim == 1:
            X_static = X_static.unsqueeze(0)
            X_time = X_time.unsqueeze(0)
            X_treat = X_treat.unsqueeze(0)
            y = y.unsqueeze(0)

        X_static_all.append(X_static)
        X_time_all.append(X_time)
        X_treat_all.append(X_treat)
        y_all.append(y)

    X_static_all = torch.cat(X_static_all, dim=0)
    X_time_all = torch.cat(X_time_all, dim=0)
    X_treat_all = torch.cat(X_treat_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    return X_static_all, X_time_all, X_treat_all, y_all

def train_eval_lr(train_subset, val_subset, args):
    model = LinearRegression()

    # train
    X_static, X_time, X_treat, y = collect_from_subset(train_subset)
    X_train = build_lr_input(X_static, X_time, X_treat).cpu().numpy()
    y_train = y[:, :, 0].reshape(-1).cpu().numpy()

    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    if args.loss == "mse" or args.loss == "pehe":
        train_loss = get_loss_value(train_preds.reshape(-1), y_train.reshape(-1), loss_type="mse")
    elif args.loss == "mae" or args.loss == "eate":
        train_loss = get_loss_value(train_preds.reshape(-1), y_train.reshape(-1), loss_type="mae")

    # val
    X_static, X_time, X_treat, y = collect_from_subset(val_subset)
    X_val = build_lr_input(X_static, X_time, X_treat).cpu().numpy()
    y_val_f = y[:, :, 0].reshape(-1).cpu().numpy()
    y_val_cf = y[:, :, 1].reshape(-1).cpu().numpy()

    val_preds = model.predict(X_val)
    # cf
    X_val_cf = build_lr_input(X_static, X_time, 1-X_treat).cpu().numpy()
    val_preds_cf = model.predict(X_val_cf)
    
    if args.loss == "mse" or args.loss == "mae":
        val_loss = get_loss_value(y_val_f, val_preds, loss_type=args.loss)
    elif args.loss == "pehe" or args.loss == "eate":
        pred_ite = val_preds - val_preds_cf
        true_ite = y_val_f - y_val_cf
        if args.loss == "pehe":
            val_loss = get_loss_value(pred_ite.reshape(-1), true_ite.reshape(-1), loss_type="mse")
        elif args.loss == "eate":
            val_loss = get_loss_value(pred_ite.reshape(-1), true_ite.reshape(-1), loss_type="mae")

    logging.info(f"LR Train: {train_loss:.6f} | LR Val: {val_loss:.6f}")
    return val_loss

def main():
    os.makedirs('log/{}'.format(date), exist_ok=True)

    args = parse_args()
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            filename='log/{}/{}.log'.format(date, hour),
            filemode='w',
            datefmt='%m/%d/%Y %I:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(args)

    ##################
    # model training #
    ##################

    device = args.device
    set_seed(seed=args.seed)
    
    dataset = torch.load(args.dataset_path)

    ##############
    # dataloader #
    ##############
    dataset = TensorDataset(dataset["statics"], dataset["feature"], dataset["treatment"], dataset["y"])
    
    ###########
    #  model  #
    ###########
    kf = KFold(n_splits=5, shuffle=True, random_state=66)
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset))), start=1):
        logging.info(f"\n========== Fold {fold} ==========")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        if args.use_smote == True:
            print(f"train size before SMOTE: {len(train_subset)}")
            
            # 畫出SMOTE後的treated/control分布圖，確認是否平衡
            train_subset_X_static, train_subset_X_time, train_subset_X_treat, train_subset_y = subset_to_tensors(train_subset)
            treat_group = train_subset_X_treat[:, 0, 0].cpu().numpy()
            treated_mask = (treat_group == 1)
            control_mask = (treat_group == 0)

            # ==========
            # counts
            # ==========
            n_treated = treated_mask.sum()
            n_control = control_mask.sum()

            print(f"Treated count: {n_treated}")
            print(f"Control count: {n_control}")

            plt.figure(figsize=(5, 4))
            plt.bar(["Control", "Treated"], [n_control, n_treated])
            plt.ylabel("Count")
            plt.title("Number of Control vs Treated Samples")
            plt.savefig("images/number_of_samples_before_smote.png")
            plt.close()
            
            # 只對 train subset 做 SMOTE
            train_subset = apply_smote_to_train_subset(
                train_subset,
                treatment_time_index=0,
                random_state=66
            )

            print(f"train size after SMOTE: {len(train_subset)}")
            print(f"Val size: {len(val_subset)}")
            
        if args.use_pca == True:
            train_subset, val_subset = apply_pca_to_fold(
                train_subset=train_subset,
                val_subset=val_subset,
                n_components_static=args.static_pca_size,
                n_components_time=args.feature_pca_size
            )
            args.feature_size = args.feature_pca_size
            args.static_size = args.static_pca_size

        if args.model == "LSTM":
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            val_loss = train_eval_lstm(
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                device=args.device,
                LSTM_model=LSTM_model
            )

        elif args.model == "LR":
            val_loss = train_eval_lr(
                train_subset=train_subset,
                val_subset=val_subset,
                args=args
            )

        all_val_losses.append(val_loss)
        logging.info(f"Fold {fold} val loss: {val_loss:.6f}")

    logging.info("\n========== CV Summary ==========")
    logging.info(f"Model: {args.model}")
    logging.info(f"Loss : {args.loss}")
    logging.info(f"Mean Val Loss: {np.mean(all_val_losses):.6f}")
    logging.info(f"Std  Val Loss: {np.std(all_val_losses):.6f}")

    return all_val_losses
        
        
if __name__ == '__main__':
    main()
