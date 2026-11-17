"""
deep_models.py
==============
Sequential Deep Learning architectures for 24-hour ahead AQI forecasting.

Includes:
1. LSTM/GRU with Temporal Self-Attention pooling.
2. Temporal Transformer Encoder.
3. Robust PyTorch training harness with Huber Loss, AdamW, and Early Stopping.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture 1: Attention-Enhanced Recurrent Forecaster (LSTM / GRU)
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Bahdanau-style temporal attention over the lookback sequence.
    Learns to dynamically weight critical past hours (e.g., rush-hour peaks or stagnant nights).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        scores = self.attn(hidden_states)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(hidden_states * weights, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class RecurrentForecaster(nn.Module):
    """
    Multi-layer LSTM or GRU with temporal attention pooling and dense MLP projection.
    Directly forecasts scalar target at horizon t+24.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: str = "lstm",
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        self.attention = TemporalAttention(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_dim)
        context, _ = self.attention(out)  # context: (batch_size, hidden_dim)
        y_pred = self.mlp(context)  # (batch_size, 1)
        return y_pred.squeeze(-1)


# ---------------------------------------------------------------------------
# Architecture 2: Temporal Transformer Encoder
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequential temporal order."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerForecaster(nn.Module):
    """
    Self-attention Transformer Encoder for multi-scale sequential dependence.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        enc = self.transformer_encoder(h)  # (batch_size, seq_len, d_model)
        # Average pooling across sequence length
        pooled = torch.mean(enc, dim=1)  # (batch_size, d_model)
        y_pred = self.head(pooled)
        return y_pred.squeeze(-1)


# ---------------------------------------------------------------------------
# Training Harness with Robust Loss and Early Stopping
# ---------------------------------------------------------------------------
class DeepLearningTrainer:
    """
    Trains PyTorch sequential models with Huber Loss, AdamW, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss for outlier robustness
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 35,
        batch_size: int = 64,
        patience: int = 7,
    ) -> Dict[str, list]:
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                running_loss += loss.item() * len(bx)

            train_loss = running_loss / len(train_dataset)

            # Validation step
            self.model.eval()
            val_running = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx)
                    loss = self.criterion(pred, by)
                    val_running += loss.item() * len(bx)

            val_loss = val_running / len(val_dataset)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 2 == 0 or epoch == epochs:
                logger.info(
                    "Epoch [%3d/%3d] Train Loss: %.4f | Val Loss: %.4f",
                    epoch, epochs, train_loss, val_loss
                )

            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return history

    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                out = self.model(bx)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)
