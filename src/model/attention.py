import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Multi-scale projections
        self.projections = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=k)
            for k in [1, 3, 5, 7]  # Multiple kernel sizes for different scales
        ])
        
        # Multi-head attention for each scale
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
            for _ in range(len(self.projections))
        ])
        
        # Scale mixing
        self.scale_mixer = nn.Parameter(torch.ones(len(self.projections)) / len(self.projections))
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Process each scale
        multi_scale_outputs = []
        for proj, attn in zip(self.projections, self.attention_layers):
            # Apply scale-specific projection
            scale_features = proj(x.transpose(1, 2)).transpose(1, 2)
            # Pad to original length if needed
            if scale_features.size(1) != seq_len:
                pad_size = seq_len - scale_features.size(1)
                scale_features = F.pad(scale_features, (0, 0, 0, pad_size))
            
            # Apply attention
            attn_output, _ = attn(scale_features, scale_features, scale_features)
            multi_scale_outputs.append(attn_output)
        
        # Mix scales
        mixed_output = sum([out * weight for out, weight in 
                          zip(multi_scale_outputs, F.softmax(self.scale_mixer, dim=0))])
        
        # Output processing
        output = self.output_proj(mixed_output)
        output = self.dropout(output)
        output = self.norm(output + x)  # Residual connection
        
        return output 