import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PressurePositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pressure):  # pressure: (B, L, 1)
        log_p = torch.log(pressure + 1e-6)  # log transform 
        return self.mlp(log_p)  # (B, L, d_model)


class CloudFractionTransformer(nn.Module):
    def __init__(
        self,
        input_dim=3,         # qci, rh, tt
        d_model=128,
        num_layers=4,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        max_levels=100
    ):
        super().__init__()
        self.d_model = d_model

        # 1.  W_embed * x_l
        self.input_fc = nn.Linear(input_dim, d_model)

        # 2. log(P_l) → MLP
        self.pressure_emb = PressurePositionalEmbedding(hidden_dim=64, output_dim=d_model)

        # 3. Learnable positional embedding 
        self.learnable_level_emb = nn.Parameter(torch.randn(max_levels, d_model))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # (B, L, d)
            norm_first=True    # LayerNorm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5.  W_out * h_l + b
        self.output_fc = nn.Linear(d_model, 1)

        # 6. Xavier 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, V, L) — V = input_dim + 1 
        """
        B, V, L = x.shape
        input_dim = self.input_fc.in_features

        x_vars = x[:, :input_dim, :]         # (B, input_dim, L)
        pressure = x[:, input_dim, :]        # (B, L)

        # 2. 
        x_vars = x_vars.permute(0, 2, 1)    # (B, L, 3)
        x_proj = self.input_fc(x_vars)     # (B, L, d_model)

        # 3.  (log(P_l) → MLP)
        pressure = pressure.unsqueeze(-1)           # (B, L, 1)
        pressure_emb = self.pressure_emb(pressure)  # (B, L, d_model)

        # 4. Learnable positional embedding
        learnable_emb = self.learnable_level_emb[:L, :].unsqueeze(0).expand(B, -1, -1)  # (B, L, d_model)

        # 5. positional embedding 
        x = x_proj + pressure_emb + learnable_emb  # (B, L, d_model)

        # 6. Transformer Encoder
        x = self.transformer_encoder(x)  # (B, L, d_model)

        # 7. 
        y_hat = self.output_fc(x).squeeze(-1)  # (B, L)

        return y_hat



