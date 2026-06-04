import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample or (in_channels != out_channels)

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=2, stride=2) if downsample else nn.Identity()
        )

        if self.downsample:
            self.skip_proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        else:
            self.skip_proj = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        skip = self.skip_proj(x)
        out = self.conv_block(x)
        return self.relu(out + skip)

class WaveformEncoder(nn.Module):
    def __init__(self, input_channels=1, input_length=150, seq_len=240, output_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.input_length = input_length

        self.front_conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(16, 32, downsample=True),
            ResBlock(32, 32)
        )

        self.layer2 = nn.Sequential(
            ResBlock(32, 64, downsample=True),
            *[ResBlock(64, 64) for _ in range(3)]
        )

        self.temporal_avg = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        # x: (B, seq_len, input_length) → e.g., (B, 240, 150)
        B, N, L = x.shape
        x = x.reshape(B, 1, N * L)             # (B, 1, 240 * L)
        out = self.front_conv(x)               # (B, 16, 240 * L)
        out = self.layer1(out)                 # (B, 32, T/2)
        out = self.layer2(out)                 # (B, 64, T/4)
        out = self.temporal_avg(out)           # (B, 64, seq_len)
        out = out.permute(0, 2, 1)             # (B, seq_len, 64)
        return out

class PreLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-Norm + Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # Pre-Norm + Feedforward
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        return x

class SleepVST(nn.Module):
    def __init__(self, seq_len=240, d_model=128, num_classes=4, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.heart_encoder = WaveformEncoder(input_channels=1, input_length=300, output_dim=64)
        self.breath_encoder = WaveformEncoder(input_channels=1, input_length=150, output_dim=64)

        # Project encodings to same dimension
        self.proj_heart = nn.Linear(64, d_model // 2)
        self.proj_breath = nn.Linear(64, d_model // 2)

        # Positional encoding (sinusoidal)
        self.positional_encoding = self._build_pos_enc(seq_len, d_model)

        # Transformer encoder with Pre-LN
        self.transformer = nn.Sequential(
            *[PreLNTransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=dropout
            ) for _ in range(num_layers)]
        )

        # Final layer normalization before classifier
        self.final_norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_hw, x_bw):
        """
        x_hw: (B, N, 300) - heart waveform
        x_bw: (B, N, 150) - breath waveform
        """
        z_hw = self.heart_encoder(x_hw)  # (B, N, 64)
        z_bw = self.breath_encoder(x_bw)  # (B, N, 64)

        # Linear projection
        z_hw = self.proj_heart(z_hw)  # (B, N, d_model/2)
        z_bw = self.proj_breath(z_bw)  # (B, N, d_model/2)

        # Concatenate along feature dim
        z = torch.cat([z_hw, z_bw], dim=-1)  # (B, N, d_model)

        # Add positional encoding
        z = z + self.positional_encoding.to(z.device)

        # Transformer encoder
        for layer in self.transformer:
            z = layer(z)  # Pre-LN 처리

        # Final layer normalization
        z = self.final_norm(z)

        # Per-epoch classification
        logits = self.classifier(z)  # (B, N, num_classes)
        return logits
    
    def forward_features(self, x_hw, x_bw):
        """
        x_hw: (B, 240, 300)
        x_bw: (B, 240, 150)
        return: (B, 240, d_model) - transformer output
        """
        z_hw = self.heart_encoder(x_hw)  # (B, 240, 64)
        z_bw = self.breath_encoder(x_bw)  # (B, 240, 64)

        z_hw = self.proj_heart(z_hw)  # (B, 240, d_model//2)
        z_bw = self.proj_breath(z_bw)  # (B, 240, d_model//2)

        z = torch.cat([z_hw, z_bw], dim=-1)  # (B, 240, d_model)
        z = z + self.positional_encoding.to(z.device)
        z_out = self.transformer(z)  # (B, 240, d_model)

        return z_out


    def _build_pos_enc(self, length, dim):
        pe = torch.zeros(length, dim)
        pos = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, length, dim)
        return pe