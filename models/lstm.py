from torch import nn

NUM_JOINTS = 13


class LSTM(nn.Module):
    def __init__(self, num_classes=15, in_channels=2, hidden=128,
                 num_layers=2, dropout=0.25):
        super().__init__()
        if min(num_classes, in_channels, hidden, num_layers) < 1:
            raise ValueError("class, channel, hidden and layer counts must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = in_channels
        # Flatten the joints and coordinates into one feature vector per frame.
        self.lstm = nn.LSTM(
            input_size=NUM_JOINTS * in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            # LSTM dropout applies between layers, so a single layer needs none.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, vis=None):
        """[N, T, 13, C] -> [N, classes] logits; state resets per clip.

        The shared loader handles invisible joints; vis is accepted for compatibility.
        """
        if x.ndim != 4 or x.shape[2:] != (NUM_JOINTS, self.in_channels):
            raise ValueError(f"expected [N, T, {NUM_JOINTS}, {self.in_channels}]")
        if x.shape[1] == 0:
            raise ValueError("clips must contain at least one frame")
        # [N, T, V, C] -> [N, T, V*C]; omitted initial states default to zeros.
        _, (hidden, _) = self.lstm(x.flatten(start_dim=2))
        # hidden[-1] is the top layer's final state; return logits for cross-entropy.
        return self.fc(self.dropout(hidden[-1]))
