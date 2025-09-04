import torch 
from torch import nn


class GRUDecoder(nn.Module):

    def __init__(self, neural_dim, n_units, n_days, n_classes,
                 rnn_dropout=0.2, input_dropout=0.1, n_layers=3,
                 bidirectional=False, patch_size=0, patch_stride=0,
                 sequence_output=True, hidden_fc=256):
        super(GRUDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.sequence_output = sequence_output
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.bidirectional = bidirectional

        # Dropout
        self.input_dropout = nn.Dropout(input_dropout)

        # More expressive day-specific frontend (2-layer MLP instead of scaling+bias)
        self.day_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(neural_dim, neural_dim),
                nn.ReLU(),
                nn.LayerNorm(neural_dim),
                nn.Linear(neural_dim, neural_dim),
            )
            for _ in range(n_days)
        ])

        # Input dimension adjustment
        self.input_size = neural_dim * (patch_size if patch_size > 0 else 1)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=n_units,
            num_layers=n_layers,
            dropout=rnn_dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = n_units * (2 if bidirectional else 1)

        # Post-GRU fully connected block
        self.fc_head = nn.Sequential(
            nn.Linear(out_dim, hidden_fc),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_fc),
            nn.Dropout(0.3),
            nn.Linear(hidden_fc, hidden_fc),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_fc),
            nn.Dropout(0.3),
        )

        # Final classification
        self.out = nn.Linear(hidden_fc, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(torch.empty(n_layers * (2 if bidirectional else 1), 1, n_units))
        nn.init.xavier_uniform_(self.h0)

        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x, day_idx, states=None, return_state=False):
        # Apply day-specific frontend
        batch_size = x.size(0)
        processed = []
        for i in range(batch_size):
            processed.append(self.day_mlp[day_idx[i]](x[i]))
        x = torch.stack(processed, dim=0)

        x = self.input_dropout(x)

        # Optional patching
        if self.patch_size > 0:
            x = x.permute(0, 2, 1)  # [B, D, T]
            patches = x.unfold(2, self.patch_size, self.patch_stride)
            x = patches.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.input_size)

        x = self.norm(x)

        # Init hidden state
        if states is None:
            states = self.h0.expand(-1, x.size(0), -1).contiguous()

        output, hidden_states = self.gru(x, states)

        # Classification
        if self.sequence_output:
            # reshape for FC head: flatten time dimension
            B, T, H = output.shape
            out_fc = self.fc_head(output.reshape(-1, H))   # [B*T, hidden_fc]
            logits = self.out(out_fc).view(B, T, -1)       # [B, T, C]
        else:
            pooled = output.mean(dim=1)                    # [B, H]
            out_fc = self.fc_head(pooled)                  # [B, hidden_fc]
            logits = self.out(out_fc)                      # [B, C]

        if return_state:
            return logits, hidden_states
        return logits

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual feedforward block: Linear -> ReLU -> BN -> Dropout -> Linear + skip"""
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D] or [B*T, D]
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return F.relu(x + residual)


class GRUDecoderAttention(nn.Module):
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 n_layers=3,
                 hidden_fc=256,
                 rnn_dropout=0.2,
                 input_dropout=0.1,
                 bidirectional=True,
                 patch_size=0,
                 patch_stride=0,
                 attn_heads=4,
                 attn_dropout=0.1,
                 n_resblocks=2):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.sequence_output = True
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.bidirectional = bidirectional

        # Dropout
        self.input_dropout = nn.Dropout(input_dropout)

        # Day embeddings (additive)
        self.day_embed = nn.Embedding(n_days, neural_dim)

        # Input dimension adjustment for patching
        self.input_size = neural_dim * (patch_size if patch_size > 0 else 1)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=n_units,
            num_layers=n_layers,
            dropout=rnn_dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = n_units * (2 if bidirectional else 1)

        # Multi-head self-attention on GRU outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Residual FC head (stacked)
        self.fc_blocks = nn.Sequential(*[
            ResidualBlock(out_dim, hidden_fc, dropout=0.3)
            for _ in range(n_resblocks)
        ])

        # Final classification
        self.out = nn.Linear(out_dim, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(torch.empty(n_layers * (2 if bidirectional else 1), 1, n_units))
        nn.init.xavier_uniform_(self.h0)

        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x: [B, T, D]
        day_idx: [B] (long tensor with day indices)
        """
        B, T, D = x.shape

        # Add day embeddings
        day_vecs = self.day_embed(day_idx)  # [B, D]
        x = x + day_vecs.unsqueeze(1)

        x = self.input_dropout(x)

        # Optional patching
        if self.patch_size > 0:
            x = x.permute(0, 2, 1)  # [B, D, T]
            patches = x.unfold(2, self.patch_size, self.patch_stride)  # [B, D, num_patches, patch_size]
            x = patches.permute(0, 2, 3, 1).reshape(B, -1, self.input_size)  # [B, num_patches, input_size]

        x = self.norm(x)

        # Init hidden state
        if states is None:
            states = self.h0.expand(-1, B, -1).contiguous()

        # GRU
        output, hidden_states = self.gru(x, states)  # [B, T, H]

        # Self-attention (contextual refinement)
        attn_out, _ = self.attn(output, output, output)  # [B, T, H]
        output = output + attn_out  # residual connection

        # Sequence or pooled output
        if self.sequence_output:
            out_fc = self.fc_blocks(output.reshape(-1, output.size(-1)))  # [B*T, H]
            logits = self.out(out_fc).view(B, -1, self.n_classes)  # [B, T, C]
        else:
            pooled = output.mean(dim=1)  # [B, H]
            out_fc = self.fc_blocks(pooled)
            logits = self.out(out_fc)  # [B, C]

        if return_state:
            return logits, hidden_states
        return logits



