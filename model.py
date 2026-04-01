import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.autograd import Function
from torch.nn import Module
from torch import tensor
import numpy as np
    
class LSTM_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.feature_embedding = nn.Linear(args.feature_size, args.emb_size)
        self.dropout = nn.Dropout(0.3)

        # Encoder uses observed window (e.g., 6 steps)
        self.lstm = nn.LSTM(args.emb_size*2 + args.treatment_dim, args.hid_size, 2, batch_first=True)
        # Static encoder
        self.static_fc = nn.Linear(args.static_size, args.emb_size)
        self.fc_out = nn.Sequential(
            nn.Linear(args.hid_size + args.treatment_dim, args.hid_size // 2),
            nn.ELU(),
            nn.Linear(args.hid_size // 2, 1)
        )
    
    def forward(
        self,
        X_static,            # [B, D_static]
        X_time,
        X_treat,             # [B, T, D_treat]
    ):
        embedded = self.feature_embedding(X_time)  # [B, T_obs, emb_dim]
        embedded = self.dropout(embedded)

        # Static embedding
        emb_static = self.static_fc(X_static)                    # [B, emb_dim]
        emb_static = emb_static.unsqueeze(1).repeat(1, X_time.shape[1], 1)  # [B, T_obs, emb_dim]
        
        # ---- Encode observed history ----
        # 在最前面補 0(對第一個時間點padding)，讓長度回到 T
        X_treat_hist = X_treat[:, :-1]
        treatment_zero = torch.zeros((X_treat.shape[0], 1, X_treat.shape[2]), device=self.device, dtype=X_treat.dtype)
        treatment_hist = torch.cat((treatment_zero, X_treat_hist), dim=1)
        
        input = torch.cat([emb_static, embedded, treatment_hist], dim=-1)  # [B, T_obs, D_feature+D_treat]
        output, (h_enc, c_enc) = self.lstm(input) 
        
        # 單點預測
        fc_input = torch.cat((output, X_treat), dim=-1)
        pred = self.fc_out(fc_input)
        return pred
