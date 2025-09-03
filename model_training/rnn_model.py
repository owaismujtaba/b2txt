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

