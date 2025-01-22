import torch.nn as nn
import torch

# Define the Decoder class
class MotionDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, unit_length):
        super(MotionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.unit_length = unit_length

        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent, lengths):
        # Expand latent vector across time steps
        batch_size = latent.shape[0]
        max_len = int(torch.max(lengths).item()) * self.unit_length  # Scale lengths by unit_length

        # Repeat latent code for the required sequence length
        latent_expanded = latent.unsqueeze(1).repeat(1, max_len, 1)  # Shape: (batch_size, max_len, latent_dim)
        
        # GRU decoding
        gru_out, _ = self.gru(latent_expanded)
        
        # Fully connected layer to map to output dimensions
        output = self.fc(gru_out)  # Shape: (batch_size, max_len, output_dim)

        return output

class MovementConvEncoderWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.3):
        super(MovementConvEncoderWithDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # Convolution
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # Down-sampling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        x = self.conv(x)  # Shape: (batch_size, hidden_dim, new_seq_len)
        x = x.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, hidden_dim)
        return x


class MotionEncoderBiGRUCoWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_prob=0.3):
        super(MotionEncoderBiGRUCoWithDropout, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_prob
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.Dropout(dropout_prob)  # Dropout before output
        )
        self.device = device

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed_x)
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        output = self.fc(unpacked_out[:, -1, :])
        return output
    
class MotionDecoderWithDropout(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, unit_length, dropout_prob=0.3):
        super(MotionDecoderWithDropout, self).__init__()
        self.unit_length = unit_length

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Dropout
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, latent, lengths):
        batch_size = latent.shape[0]
        max_len = int(torch.max(lengths).item()) * self.unit_length

        latent_expanded = latent.unsqueeze(1).repeat(1, max_len, 1)
        gru_out, _ = self.gru(latent_expanded)
        output = self.fc(gru_out)
        return output