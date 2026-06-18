"""SleepVST: a Transformer-based sleep staging model from cardiorespiratory waveforms.

This module implements the SleepVST architecture, which classifies sleep stages
from heart (ECG/PPG) and breathing waveforms. The pipeline is:

    1. A 1D ResNet-style ``WaveformEncoder`` turns each modality's raw waveform
       into a sequence of per-epoch feature vectors.
    2. The encoded modalities are linearly projected and concatenated into a
       single token sequence, then augmented with sinusoidal positional encoding.
    3. A stack of Pre-LN Transformer encoder layers contextualizes the sequence.
    4. A linear classifier produces per-epoch sleep-stage logits.

Two variants are provided:
    * :class:`SleepVST`    -- uses both heart and breath waveforms.
    * :class:`SleepVST_BW` -- uses the breath waveform only.
"""

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """A 1D residual block with optional temporal downsampling.

    Applies two ``Conv1d -> BatchNorm`` layers (the first followed by ReLU) on the
    main path and adds a skip connection. When ``downsample`` is requested -- or
    when the channel count changes -- the temporal dimension is halved via
    ``MaxPool1d`` and the skip path is projected with a 1x1 convolution so the
    shapes match before the final ReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample (bool): If ``True``, halve the temporal length. Downsampling is
            also forced whenever ``in_channels != out_channels``.
    """

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
        """Run the residual block.

        Args:
            x (torch.Tensor): Input of shape ``(B, in_channels, T)``.

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, T')`` where ``T'`` is
            ``T // 2`` when downsampling, otherwise ``T``.
        """
        skip = self.skip_proj(x)
        out = self.conv_block(x)
        return self.relu(out + skip)

class WaveformEncoder(nn.Module):
    """1D ResNet encoder that maps a raw waveform to per-epoch feature vectors.

    The full waveform of all epochs is flattened into a single 1D signal, passed
    through a front convolution and two residual stages (each stage downsamples
    the temporal dimension), then adaptively pooled back to ``seq_len`` time steps
    -- one feature vector per epoch.

    Args:
        input_channels (int): Number of input waveform channels (typically 1).
        input_length (int): Number of samples per epoch.
        seq_len (int): Number of epochs in the sequence; the output length.
        output_dim (int): Feature dimension of each per-epoch output vector.
    """

    def __init__(self, input_channels=1, input_length=150, seq_len=240, output_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.input_length = input_length
        self.output_dim = output_dim

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
            ResBlock(32, output_dim, downsample=True),  # 64 대신 output_dim 사용
            *[ResBlock(output_dim, output_dim) for _ in range(3)]
        )

        self.temporal_avg = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        """Encode a batch of per-epoch waveforms.

        Args:
            x (torch.Tensor): Input of shape ``(B, N, input_length)`` where ``N`` is
                the number of epochs.

        Returns:
            torch.Tensor: Encoded features of shape ``(B, seq_len, output_dim)``.
        """
        # x: (B, seq_len, input_length) → e.g., (B, 240, 150)
        B, N, L = x.shape
        x = x.reshape(B, 1, N * L)             # (B, 1, 240 * L)
        out = self.front_conv(x)               # (B, 16, 240 * L)
        out = self.layer1(out)                 # (B, 32, T/2)
        out = self.layer2(out)                 # (B, output_dim, T/4)
        out = self.temporal_avg(out)           # (B, output_dim, seq_len)
        out = out.permute(0, 2, 1)             # (B, seq_len, output_dim)
        return out

class PreLNTransformerEncoderLayer(nn.Module):
    """A Pre-LayerNorm Transformer encoder layer.

    Unlike the post-norm layer in ``nn.TransformerEncoderLayer``, normalization is
    applied *before* the self-attention and feed-forward sub-blocks, with residual
    connections wrapping each. Pre-LN improves training stability for deep stacks.

    Args:
        d_model (int): Model (embedding) dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Hidden dimension of the feed-forward sub-block.
        dropout (float): Dropout probability used in attention and feed-forward.
    """

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
        """Apply Pre-LN self-attention and feed-forward sub-blocks.

        Args:
            x (torch.Tensor): Input of shape ``(B, seq_len, d_model)``.

        Returns:
            torch.Tensor: Output of the same shape ``(B, seq_len, d_model)``.
        """
        # Pre-Norm + Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # Pre-Norm + Feedforward
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        return x

class SleepVST(nn.Module):
    """SleepVST sleep-stage classifier using heart and breath waveforms.

    Each modality is encoded by its own :class:`WaveformEncoder`, projected to
    ``d_model // 2``, and concatenated into ``d_model``-dimensional tokens. After
    adding sinusoidal positional encoding, a stack of Pre-LN Transformer layers
    contextualizes the sequence and a linear head emits per-epoch logits.

    Hydra instantiates this class from the model yaml by passing config entries
    directly as keyword arguments.

    Args:
        seq_len (int): Number of epochs per sequence.
        d_model (int): Transformer model dimension.
        num_classes (int): Number of sleep-stage classes.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        input_length (int): Number of samples in one breath waveform epoch.
        **kwargs: Non-architecture metadata from the model yaml, such as
            ``name`` and checkpoint paths.
    """

    def __init__(self, seq_len=240, d_model=128, num_classes=4, num_layers=6,
                 num_heads=8, dropout=0.1, input_length=150, **kwargs):
        super().__init__()

        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.input_length = int(input_length)

        self.heart_encoder = WaveformEncoder(input_channels=1, input_length=300, output_dim=64)
        self.breath_encoder = WaveformEncoder(input_channels=1, input_length=self.input_length, output_dim=64)

        # Project encodings to same dimension
        self.proj_heart = nn.Linear(64, self.d_model // 2)
        self.proj_breath = nn.Linear(64, self.d_model // 2)

        # Positional encoding (sinusoidal)
        self.positional_encoding = self._build_pos_enc(self.seq_len, self.d_model)

        # Transformer encoder with Pre-LN
        self.transformer = nn.Sequential(
            *[PreLNTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=512,
                dropout=self.dropout
            ) for _ in range(self.num_layers)]
        )

        # Final layer normalization before classifier
        self.final_norm = nn.LayerNorm(self.d_model)

        # Classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x_hw, x_bw):
        """Classify sleep stages from heart and breath waveforms.

        Args:
            x_hw (torch.Tensor): Heart waveform of shape ``(B, N, 300)``.
            x_bw (torch.Tensor): Breath waveform of shape ``(B, N, 150)``.

        Returns:
            torch.Tensor: Per-epoch logits of shape ``(B, N, num_classes)``.
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
        """Return the Transformer features without the classification head.

        Useful for representation learning, probing, or downstream fine-tuning.
        Note that, unlike :meth:`forward`, the final LayerNorm is not applied.

        Args:
            x_hw (torch.Tensor): Heart waveform of shape ``(B, 240, 300)``.
            x_bw (torch.Tensor): Breath waveform of shape ``(B, 240, 150)``.

        Returns:
            torch.Tensor: Transformer output of shape ``(B, 240, d_model)``.
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
        """Build a fixed sinusoidal positional encoding.

        Args:
            length (int): Sequence length (number of positions).
            dim (int): Encoding dimension; must be even.

        Returns:
            torch.Tensor: Positional encoding of shape ``(1, length, dim)``.
        """
        pe = torch.zeros(length, dim)
        pos = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, length, dim)
        return pe

class SleepVST_BW(SleepVST):
    """Breath-only variant of :class:`SleepVST`.

    Uses a single :class:`WaveformEncoder` on the breathing waveform (with
    ``output_dim == d_model``) and skips the heart branch. It reuses the parent's
    positional-encoding builder and overrides :meth:`forward` to take only the
    breath waveform.

    Args:
        seq_len (int): Number of epochs per sequence.
        d_model (int): Transformer model dimension.
        num_classes (int): Number of sleep-stage classes.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        input_length (int): Number of samples in one breath waveform epoch.
        **kwargs: Non-architecture metadata from the model yaml, such as
            ``name`` and checkpoint paths.
    """

    def __init__(self, seq_len=240, d_model=128, num_classes=4, num_layers=6,
                 num_heads=8, dropout=0.1, input_length=150, **kwargs):
        nn.Module.__init__(self)

        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.input_length = int(input_length)

        # BW만 사용하는 인코더
        self.breath_encoder = WaveformEncoder(input_channels=1, input_length=self.input_length, output_dim=128)
        self.proj_breath = nn.Linear(128, self.d_model)

        # Positional encoding
        self.positional_encoding = self._build_pos_enc(self.seq_len, self.d_model)

        # Transformer encoder with Pre-LN
        self.transformer = nn.Sequential(
            *[PreLNTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=512,
                dropout=self.dropout
            ) for _ in range(self.num_layers)]
        )

        # Final layer normalization before classifier
        self.final_norm = nn.LayerNorm(self.d_model)

        # Classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x_bw):
        """Classify sleep stages from the breath waveform alone.

        Args:
            x_bw (torch.Tensor): Breath waveform of shape ``(B, N, 150)``.

        Returns:
            torch.Tensor: Per-epoch logits of shape ``(B, N, num_classes)``.
        """
        z_bw = self.breath_encoder(x_bw)  # (B, N, 128)
        z = self.proj_breath(z_bw)  # (B, N, d_model)

        # Add positional encoding
        z = z + self.positional_encoding.to(z.device)

        # Transformer encoder
        for layer in self.transformer:
            z = layer(z)

        # Final layer normalization
        z = self.final_norm(z)

        # Per-epoch classification
        logits = self.classifier(z)  # (B, N, num_classes)
        return logits