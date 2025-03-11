import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Sorry Andrew, messy first dump

"""
A hierarchical transformer architecture for analyzing and classifying genomic sequences.

This model processes genomic data at multiple levels:
1. K-mer level: Captures local nucleotide patterns using variable-sized k-mers
2. Contig level: Processes individual DNA sequence fragments using transformers
3. Assembly level: Integrates information across multiple contigs

Key Features:
- Multi-scale k-mer embeddings to capture patterns at different resolutions
- Hierarchical transformer architecture for processing sequence fragments and whole assemblies
- Custom attention pooling to focus on the most informative regions
- Integration of assembly-level features

The Application here would be to do taxnomic classification
"""


# I actually messed up here and need to add or remove the assembly_features, this would come from
# Probably don't want these in data loading class where we currently have it
# The input would come from the assembly csv itself - need to think about this
@dataclass
class ModelConfig:
    """Configuration for the Genomic Transformer Model."""
    max_contigs: int = 200
    max_seq_len: int = 1000
    kmer_sizes: List[int] = None
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    num_classes: int = 10
    dropout: float = 0.1
    assembly_features: int = 4

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    # For now using this positional encoding, but will try with GenomicPositionalEncoding down the line
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer 
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
        """
        # Safety check for tensor dimensions
        if len(x.shape) != 3:
            # If input is not 3D, reshape it to be compatible
            if len(x.shape) == 2:
                # Assume [batch_size, embedding_dim]
                x = x.unsqueeze(1)  # Make it [batch_size, 1, embedding_dim]
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
        
        # Get seq_len and make sure it doesn't exceed our pe buffer
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            seq_len = self.pe.size(1)
            x = x[:, :seq_len, :]
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        return x


class GenomicPositionalEncoding(nn.Module):
    """Genomic-specific positional encoding with the option of learned positions."""
    
    # This should help preserve the sequential nature of genomes down the line
    # As transformers are "position blind"
    def __init__(self, d_model, max_len=1000, learned=False):
        super().__init__()
        
        if learned:
            # Learned positional embeddings
            self.position_embeddings = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
        else:
            # Standard sinusoidal encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        self.learned = learned
        
    def forward(self, x, positions=None):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
            positions: Optional custom positions [batch_size, seq_len]
                      (e.g., genomic coordinates rather than sequence indices)
        """
        if self.learned:
            # Use standard indices if positions not provided
            if positions is None:
                positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
            
            position_embeddings = self.position_embeddings[positions]
            return x + position_embeddings
        else:
            return x + self.pe[:, :x.size(1), :]


class AttentionPooling(nn.Module):
    """Self-attention pooling layer to aggregate sequences."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Apply attention pooling to sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask of shape [batch_size, seq_len]
            
        Returns:
            Pooled tensor of shape [batch_size, hidden_dim]
        """
        # Calculate attention scores
        scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Apply attention weights
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch_size, hidden_dim]
        
        return pooled


class TransformerEncoder(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Apply transformer encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask of shape [seq_len]
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        # For PyTorch's MultiheadAttention:
        # - The dimensions must match exactly for the key_padding_mask
        # - key_padding_mask should be (batch_size, src_len)
        
        # Simplify: just don't use key_padding_mask for now to debug other issues
        key_padding_mask = None
        
        attn_output, _ = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class GenomicTransformerV2(nn.Module):
    """Improved transformer model for genomic sequence classification with positional encoding."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Set default k-mer sizes if not provided
        if self.config.kmer_sizes is None:
            self.config.kmer_sizes = [4, 6, 8]
        
        # Create k-mer embeddings for each k-mer size
        self.kmer_embeddings = nn.ModuleList([
            nn.Embedding(4**k, config.embedding_dim // len(config.kmer_sizes))
            for k in config.kmer_sizes
        ])
        
        # Projection layer to combine k-mer embeddings
        self.projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        # Add positional encodings for sequences and contigs
        self.seq_positional_encoding = PositionalEncoding(
            config.hidden_dim, max_len=config.max_seq_len
        )
        self.contig_positional_encoding = PositionalEncoding(
            config.hidden_dim, max_len=config.max_contigs
        )
        
        # Assembly features encoder
        self.assembly_encoder = nn.Sequential(
            nn.Linear(config.assembly_features, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Contig-level transformer
        self.contig_transformer = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers // 2)
        ])
        
        # Contig pooling
        self.contig_pooling = AttentionPooling(config.hidden_dim)
        
        # Assembly-level transformer
        self.assembly_transformer = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers // 2)
        ])
        
        # Assembly pooling
        self.assembly_pooling = AttentionPooling(config.hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(
        self,
        kmer_indices: List[torch.Tensor],
        padding_mask: torch.Tensor,
        assembly_features: torch.Tensor
    ):
        """Simplified forward pass for debugging."""
        batch_size, max_contigs = padding_mask.shape
        
        # Get embedded k-mers for each contig
        contig_embeddings = []
        for b in range(batch_size):
            assembly_contigs = []
            for c in range(max_contigs):
                if padding_mask[b, c]:
                    # Embed k-mers for this contig
                    k_embeddings = []
                    for k_idx, indices in enumerate(kmer_indices):
                        if k_idx >= len(self.kmer_embeddings):
                            continue
                        
                        # Get valid indices
                        valid_indices = (indices[b, c] != 0)
                        if valid_indices.any():
                            # Embed k-mers
                            k_emb = self.kmer_embeddings[k_idx](indices[b, c][valid_indices])
                            # Pool k-mer embeddings
                            k_emb = k_emb.mean(dim=0)
                            k_embeddings.append(k_emb)
                    
                    # Combine k-mer embeddings if we have any
                    if k_embeddings:
                        contig_emb = torch.cat(k_embeddings)
                        assembly_contigs.append(contig_emb)
            
            # If we have contigs, add them to the batch
            if assembly_contigs:
                # Pad to same length
                max_len = max(emb.shape[0] for emb in assembly_contigs)
                padded_contigs = []
                for emb in assembly_contigs:
                    if emb.shape[0] < max_len:
                        pad = torch.zeros(max_len - emb.shape[0], device=emb.device)
                        emb = torch.cat([emb, pad])
                    padded_contigs.append(emb)
                contig_embeddings.append(torch.stack(padded_contigs))
            else:
                # If no contigs, add a dummy embedding
                contig_embeddings.append(torch.zeros(1, self.config.embedding_dim, device=assembly_features.device))
        
        # Project to hidden dimension
        contig_vectors = []
        for b in range(batch_size):
            contig_emb = contig_embeddings[b]
            
            # Project to hidden dimension
            contig_emb = self.projection(contig_emb)
            
            # Apply positional encoding
            contig_emb = self.seq_positional_encoding(contig_emb)
            
            contig_vectors.append(contig_emb)
        
        # Process each assembly with transformers
        assembly_vectors = []
        for b in range(batch_size):
            # Get contig embeddings for this assembly
            contig_emb = contig_vectors[b]
            
            # Apply contig transformer without mask
            for layer in self.contig_transformer:
                contig_emb = layer(contig_emb) 
            
            # Pool contigs - use mean along first dimension
            assembly_emb = contig_emb.mean(dim=0)
            assembly_vectors.append(assembly_emb)
        
        # Stack assembly vectors - this should give [batch_size, hidden_dim]
        # Have to track the dimensions everywhere in PyTorch...Sigh
        assembly_emb = torch.stack(assembly_vectors)  
        
        # Print shapes for debugging
        print(f"assembly_emb shape: {assembly_emb.shape}")
        print(f"assembly_features shape: {assembly_features.shape}")
        
        # Process assembly features
        assembly_feat_emb = self.assembly_encoder(assembly_features)
        print(f"assembly_feat_emb shape: {assembly_feat_emb.shape}")
        
        # Make sure dimensions match before concatenating
        # If assembly_emb is 3D and assembly_feat_emb is 2D, fix the dimensions
        if len(assembly_emb.shape) == 3 and len(assembly_feat_emb.shape) == 2:
            # Either squeeze assembly_emb if it has a singleton dimension
            if assembly_emb.shape[1] == 1:
                assembly_emb = assembly_emb.squeeze(1)
            # Or unsqueeze assembly_feat_emb
            else:
                assembly_feat_emb = assembly_feat_emb.unsqueeze(1)
        
        # Combine with assembly features - now with matching dimensions
        combined = torch.cat([assembly_emb, assembly_feat_emb], dim=-1)
        
        # Apply classifier
        logits = self.classifier(combined)
        
        return logits
    
# Function to create a model from config
def create_model(config: ModelConfig):
    """Create a genomic transformer model from config."""
    model = GenomicTransformerV2(config)
    return model


if __name__ == "__main__":
    # Create model config
    config = ModelConfig(
        max_contigs=50,
        max_seq_len=500,
        kmer_sizes=[4, 6, 8],
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        num_classes=5,
        dropout=0.1,
        assembly_features=4
    )
    
    model = create_model(config)
    
    print(model)
    
    # Create dummy input
    batch_size = 2
    kmer_indices = [
        torch.randint(0, 4**4, (batch_size, config.max_contigs, config.max_seq_len)),
        torch.randint(0, 4**6, (batch_size, config.max_contigs, config.max_seq_len)),
        torch.randint(0, 4**8, (batch_size, config.max_contigs, config.max_seq_len))
    ]
    padding_mask = torch.ones(batch_size, config.max_contigs)
    assembly_features = torch.rand(batch_size, config.assembly_features)
    
    # Forward pass
    logits = model(kmer_indices, padding_mask, assembly_features)
    print(f"Output shape: {logits.shape}")