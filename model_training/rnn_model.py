import torch 
from torch import nn

import torch
from torch import nn


class GRUDecoder(nn.Module):
    def __init__(self, neural_dim, n_units, n_days, n_classes,
                 rnn_dropout=0.2, input_dropout=0.1, n_layers=3,
                 bidirectional=False, patch_size=0, patch_stride=0,
                 sequence_output=True, hidden_fc=256):
        super().__init__()

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

class GRUDecoder1(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits
        

