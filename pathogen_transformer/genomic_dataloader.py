import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

# Could even remove this and use something from HuggingFace Transformers library to embed kmers 
def create_kmer_indices(sequence: str, k: int) -> torch.Tensor:
    """Convert a DNA sequence to k-mer indices with improved handling of non-standard bases."""
    # Map nucleotides to integers
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Non 0 - 3 value for N's. Not sure if this is the best way to handle non standard tokens, but 
    # For now this will do
    non_standard_base_start = 4**k
    
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        
        # Check if kmer contains non-standard bases
        if any(base not in base_to_idx for base in kmer):
            # Add non-standard tokens
            kmers.append(non_standard_base_start)
        else:
            # Convert k-mer to unique index by treating ACGT as digits in a base-4 numeral system
            idx = 0
            for j, base in enumerate(kmer):
                idx += base_to_idx[base] * (4 ** (k - j - 1))
            kmers.append(idx)
    
    # Return tensor of k-mer indices
    if not kmers:
        return torch.zeros(1, dtype=torch.long)  # Return at least one index if empty
    return torch.tensor(kmers, dtype=torch.long)

# Eventually this will want to get expanded to probably include other assembly features
class SimplePathogenDataset(Dataset):
    """Dataset for pathogen classification using assembly files."""
    
    def __init__(
        self, 
        csv_file: str,
        max_contigs: int = 200,
        max_contig_len: int = 1000,
        kmer_sizes: List[int] = [4, 6, 8],
        min_contig_len: int = 500,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with columns 'species' and 'assembly_path'
            max_contigs: Maximum number of contigs to include per assembly
            max_contig_len: Maximum length of each contig to process
            kmer_sizes: List of k-mer sizes to use
            min_contig_len: Minimum contig length to include
            cache_dir: Directory to cache processed data (None for no caching)
        """
        self.metadata = pd.read_csv(csv_file)
        self.max_contigs = max_contigs
        self.max_contig_len = max_contig_len
        self.kmer_sizes = kmer_sizes
        self.min_contig_len = min_contig_len
        print("Initializing class")
        
        # Create label mapping
        self.species = sorted(self.metadata['species'].unique())
        self.species_to_idx = {sp: i for i, sp in enumerate(self.species)}
        
        # Set up cache
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a processed assembly by index."""
        row = self.metadata.iloc[idx]
        species = row['species']
        assembly_path = row['assembly_path']
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(assembly_path).stem}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        
        # Process the assembly
        result = self._process_assembly(assembly_path, species)
        
        # Cache the result
        if self.cache_dir:
            torch.save(result, cache_path)
        
        return result
    
    def _process_assembly(self, assembly_path, species):
        """Process a single assembly file."""
        # Read assembly file
        contigs = []
        for record in SeqIO.parse(assembly_path, "fasta"):
            if len(record.seq) >= self.min_contig_len:
                contigs.append(str(record.seq))
        
        # Sort contigs by length (descending)
        contigs.sort(key=len, reverse=True)
        
        # Select top contigs - I think we want to do this
        contigs = contigs[:self.max_contigs]
        n_contigs = len(contigs)
        
        # Initialize tensors for k-mer indices
        kmer_indices_list = []
        for k in self.kmer_sizes:
            indices_list = []
            for contig in contigs:
                # Truncate contig if needed
                contig_seq = contig[:self.max_contig_len]
                # Get k-mer indices
                indices = create_kmer_indices(contig_seq, k)
                indices_list.append(indices)
            kmer_indices_list.append(indices_list)
        
        # Create padding mask
        padding_mask = torch.zeros(self.max_contigs, dtype=torch.bool)
        padding_mask[:n_contigs] = 1
        
        # Get label
        label = torch.tensor(self.species_to_idx[species], dtype=torch.long)
        
        # Calculate basic assembly features - just for example
        # Didn't check if these are correct or not
        total_length = sum(len(contig) for contig in contigs)
        n50 = 0
        cumulative = 0
        for length in sorted([len(contig) for contig in contigs], reverse=True):
            cumulative += length
            if cumulative >= total_length / 2:
                n50 = length
                break
        
        gc_content = sum(contig.count('G') + contig.count('C') for contig in contigs) / total_length if total_length > 0 else 0
        
        # Create assembly features tensor
        assembly_features = torch.tensor([
            n_contigs / self.max_contigs,  # Normalized number of contigs
            total_length / 1e6,  # Total length in Mb
            n50 / 1e5,  # N50 in units of 100kb
            gc_content  # GC content
        ], dtype=torch.float)
        
        return {
            'kmer_indices': kmer_indices_list,
            'padding_mask': padding_mask,
            'label': label,
            'assembly_features': assembly_features,
            'species': species
        }


    def collate_fn(self,batch):
        """Collate function for batching."""
        # Get batch size and k-mer sizes
        batch_size = len(batch)
        kmer_sizes = len(batch[0]['kmer_indices'])
        max_contigs = batch[0]['padding_mask'].size(0)
        
        # Get the maximum sequence length for each k-mer size and contig
        max_lengths = []
        for k_idx in range(kmer_sizes):
            contig_max_lengths = []
            for b_idx in range(batch_size):
                contig_lengths = [indices.size(0) for indices in batch[b_idx]['kmer_indices'][k_idx]]
                # Pad with zeros for missing contigs
                contig_lengths.extend([0] * (max_contigs - len(contig_lengths)))
                contig_max_lengths.append(contig_lengths)
            
            # Get max length for each contig position
            max_lengths.append([max(lengths) for lengths in zip(*contig_max_lengths)])
        
        # Initialize output tensors
        kmer_indices = []
        for k_idx in range(kmer_sizes):
            # Create tensor for this k-mer size
            indices_tensor = torch.zeros(batch_size, max_contigs, max(1, max(max_lengths[k_idx])), dtype=torch.long)
            
            # Fill tensor with k-mer indices
            for b_idx in range(batch_size):
                for c_idx, indices in enumerate(batch[b_idx]['kmer_indices'][k_idx]):
                    if indices.size(0) > 0:
                        indices_tensor[b_idx, c_idx, :indices.size(0)] = indices
            
            kmer_indices.append(indices_tensor)
        
        # Stack padding masks and labels
        padding_mask = torch.stack([item['padding_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Stack assembly features
        assembly_features = torch.stack([item['assembly_features'] for item in batch])
        
        # Collect species names
        species = [item['species'] for item in batch]
        
        return {
            'kmer_indices': kmer_indices,
            'padding_mask': padding_mask,
            'label': labels,
            'assembly_features': assembly_features,
            'species': species
        }


# Example usage
def test_dataloader():
    
    train_csv = "test_data/test_data.csv"
    
    # Create dataset
    dataset = SimplePathogenDataset(
        csv_file=train_csv,
        max_contigs=50,
        max_contig_len=500,
        kmer_sizes=[4, 6],
        cache_dir="cache"
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Species mapping: {dataset.species_to_idx}")
    
    # Test iteration
    for batch in dataloader:
        print(f"Batch shapes:")
        for k_idx, indices in enumerate(batch['kmer_indices']):
            print(f"  k-mer size {dataset.kmer_sizes[k_idx]}: {indices.shape}")
        print(f"  Padding mask: {batch['padding_mask'].shape}")
        print(f"  Labels: {batch['label'].shape}")
        print(f"  Assembly features: {batch['assembly_features'].shape}")
        print(f"  Species: {batch['species']}")
        break


if __name__ == "__main__":
    print("test")
    test_dataloader()