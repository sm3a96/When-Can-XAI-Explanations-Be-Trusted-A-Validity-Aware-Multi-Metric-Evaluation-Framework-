"""
model_definitions.py — Shared model class definitions
======================================================
Used by both training scripts and Generate_Explanations.py.
Avoids circular import issues when loading saved .pth files.
"""

import torch
import torch.nn as nn
import numpy as np


class FeatureTokenizerTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.1, **_):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        self.input_norm = nn.BatchNorm1d(n_features)
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model))
            for _ in range(n_features)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.classifier  = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.last_attn_weights = None

    def forward(self, x):
        B  = x.size(0)
        x  = self.input_norm(x)
        tokens = torch.stack([emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)], dim=1)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out    = self.norm(self.transformer(tokens))
        return self.classifier(out[:, 0, :])

    @torch.no_grad()
    def get_attention_weights(self, x):
        B  = x.size(0)
        x  = self.input_norm(x)
        tokens = torch.stack([emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)], dim=1)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        attn_list = []
        src = tokens
        for layer in self.transformer.layers:
            _, w = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
            attn_list.append(w[:, 0, 1:].detach().cpu().numpy())
            src = layer(src)
        return np.stack(attn_list, axis=0).mean(0)  # (B, n_features)


class FeatureAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super().__init__()
        h = hidden_size * 2 if bidirectional else hidden_size
        self.attn = nn.Linear(h, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * lstm_out).sum(1)
        return context, weights


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=64, n_layers=2,
                 dropout=0.3, bidirectional=True, **_):
        super().__init__()
        self.hidden_size  = hidden_size
        self.bidirectional = bidirectional
        self.input_norm   = nn.BatchNorm1d(n_features)
        self.lstm         = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional, batch_first=True,
        )
        self.attention    = FeatureAttention(hidden_size, bidirectional)
        h_out             = hidden_size * 2 if bidirectional else hidden_size
        self.classifier   = nn.Sequential(
            nn.LayerNorm(h_out), nn.Dropout(dropout), nn.Linear(h_out, n_classes)
        )
        self.last_attn_weights = None

    def forward(self, x):
        x   = self.input_norm(x)
        out, _ = self.lstm(x.unsqueeze(-1))
        ctx, w = self.attention(out)
        self.last_attn_weights = w.detach()
        return self.classifier(ctx)

    @torch.no_grad()
    def get_attention_weights(self, x):
        x   = self.input_norm(x)
        out, _ = self.lstm(x.unsqueeze(-1))
        _, w   = self.attention(out)
        return w.cpu().numpy()


def load_model(pth_path: str, device: torch.device):
    """Load any saved .pth checkpoint and return (model, feature_names, label_classes)."""
    ckpt   = torch.load(pth_path, map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    feats  = ckpt["feature_names"]
    classes = list(ckpt["label_encoder_classes"])

    if "d_model" in cfg:
        model = FeatureTokenizerTransformer(
            n_features=len(feats), n_classes=len(classes),
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"]
        )
    else:
        model = LSTMClassifier(
            n_features=len(feats), n_classes=len(classes),
            hidden_size=cfg["hidden_size"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", True)
        )

    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, feats, classes
