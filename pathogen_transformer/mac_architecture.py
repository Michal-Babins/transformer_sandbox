import os
import torch
import platform
from model import ModelConfig, create_model
from trainer import GenomicTransformerTrainer, TrainingConfig
from genomic_dataloader import SimplePathogenDataset
from torch.utils.data import DataLoader


class MacTestingConfig:
    """Configuration specifically for testing on Mac M3 hardware."""
    def __init__(
        self,
        output_dir: str = "mac_test_outputs",
        use_mps: bool = True,
        memory_efficient: bool = True,
        batch_size: int = 4,
        test_epochs: int = 3,
        gradient_checkpointing: bool = True,
        small_model: bool = True,
        profiling: bool = True
    ):
        """
        Args:
            output_dir: Directory to save outputs
            use_mps: Whether to use MPS (Metal Performance Shaders) for Mac GPU
            memory_efficient: Enable memory efficiency optimizations
            batch_size: Small batch size appropriate for testing
            test_epochs: Number of epochs for quick testing
            gradient_checkpointing: Use gradient checkpointing to save memory
            small_model: Use a smaller model for testing
            profiling: Whether to enable performance profiling
        """
        self.output_dir = output_dir
        self.use_mps = use_mps
        self.memory_efficient = memory_efficient
        self.batch_size = batch_size
        self.test_epochs = test_epochs
        self.gradient_checkpointing = gradient_checkpointing
        self.small_model = small_model
        self.profiling = profiling


def is_mac_silicon():
    """Check if running on Mac with Apple Silicon."""
    return (
        platform.system() == "Darwin" and 
        platform.machine() == "arm64"
    )


def get_optimal_device():
    """Get the optimal device for Mac testing."""
    if is_mac_silicon() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_small_model_config():
    """Create a small model configuration suitable for Mac testing."""
    return ModelConfig(
        max_contigs=20,  # Reduced from 50
        max_seq_len=200,  # Reduced from 500
        kmer_sizes=[4, 6],  # Removed k=8 to reduce parameters
        embedding_dim=64,  # Reduced from 128
        hidden_dim=128,  # Reduced from 256
        num_heads=2,  # Reduced from 4
        num_layers=2,  # Reduced from 4
        num_classes=5,  # Adjust according to your dataset
        dropout=0.1,
        assembly_features=4
    )


def get_mac_training_config(mac_config: MacTestingConfig):
    """Create a training configuration optimized for Mac testing."""
    return TrainingConfig(
        output_dir=mac_config.output_dir,
        batch_size=mac_config.batch_size,
        eval_batch_size=mac_config.batch_size,
        num_epochs=mac_config.test_epochs,
        learning_rate=3e-4,
        weight_decay=0.01,
        early_stopping_patience=2,  # Reduced patience for faster testing
        logging_steps=10,  # More frequent logging for testing
        save_steps=50,  # Less frequent saves to avoid IO overhead
        save_total_limit=1,  # Only keep the best model
        scheduler_type="cosine",
        use_mixed_precision=False,  # Disable mixed precision on MPS for now
        gradient_accumulation_steps=2  # Use gradient accumulation for effective batch size
    )


def create_small_test_dataset(
    input_csv: str, 
    output_csv: str, 
    sample_size: int = 50
):
    """
    Create a small test dataset from a larger one.
    
    Args:
        input_csv: Path to the original CSV file
        output_csv: Path to save the small test CSV
        sample_size: Number of samples to include
    """
    import pandas as pd
    
    # Read the original CSV
    df = pd.read_csv(input_csv)
    
    # Get a stratified sample if possible
    if 'species' in df.columns:
        # Get a balanced sample per species
        samples = []
        for species in df['species'].unique():
            species_df = df[df['species'] == species]
            species_sample_size = min(
                sample_size // len(df['species'].unique()), 
                len(species_df)
            )
            samples.append(species_df.sample(species_sample_size))
        small_df = pd.concat(samples)
    else:
        # Just take a random sample
        small_df = df.sample(min(sample_size, len(df)))
    
    # Save to output CSV
    small_df.to_csv(output_csv, index=False)
    
    print(f"Created small test dataset with {len(small_df)} samples at {output_csv}")
    return output_csv


class MemoryEfficientLoader:
    """A wrapper for DataLoader that implements memory-efficient loading strategies."""
    
    def __init__(self, dataloader, prefetch=2, pin_memory=False):
        """
        Args:
            dataloader: Original DataLoader
            prefetch: Number of batches to prefetch
            pin_memory: Whether to pin memory
        """
        self.dataloader = dataloader
        self.prefetch = prefetch
        
        # Disable pinned memory on Mac (can cause issues)
        if is_mac_silicon():
            pin_memory = False
        
        # Update dataloader settings
        self.dataloader.pin_memory = pin_memory
    
    def __iter__(self):
        """Return an efficient iterator over batches."""
        # Use a simple iterator with minimal caching
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def dataset(self):
        return self._dataset
        
    def __getattr__(self, name):
        return getattr(self.dataloader, name)

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing on the model to save memory.
    
    Args:
        model: The GenomicTransformerV2 model
    """
    # Add gradient checkpointing to transformer layers
    for layer in model.contig_transformer:
        if hasattr(layer.self_attn, "checkpoint") and callable(layer.self_attn.checkpoint):
            layer.self_attn.checkpoint = True
    
    for layer in model.assembly_transformer:
        if hasattr(layer.self_attn, "checkpoint") and callable(layer.self_attn.checkpoint):
            layer.self_attn.checkpoint = True
    
    # Return the modified model
    return model


def run_mac_testing(
    mac_config: MacTestingConfig,
    train_dataloader,
    val_dataloader,
    test_dataloader=None,
    class_names=None
):
    """
    Run a memory-efficient test on Mac hardware.
    
    Args:
        mac_config: Configuration for Mac testing
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Test data loader (optional)
        class_names: Class names for reporting (optional)
    
    Returns:
        The trained model and training history
    """
    # Set the optimal device
    device = get_optimal_device()
    if mac_config.use_mps and device.type != "mps" and is_mac_silicon():
        print("Warning: MPS requested but not available. Using CPU instead.")
    
    print(f"Running with device: {device}")
    
    # Create memory-efficient dataloaders
    if mac_config.memory_efficient:
        train_dataloader = MemoryEfficientLoader(train_dataloader)
        val_dataloader = MemoryEfficientLoader(val_dataloader)
        if test_dataloader:
            test_dataloader = MemoryEfficientLoader(test_dataloader)
    
    # Create model
    if mac_config.small_model:
        model_config = get_small_model_config()
        print("Using small model configuration for Mac testing")
    else:
        # Use the default model config from your script
        raise ValueError("Please provide a model configuration for full-sized model testing")
    
    model = create_model(model_config)
    
    # Enable gradient checkpointing if requested
    if mac_config.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")
    
    # Create training config
    training_config = get_mac_training_config(mac_config)
    
    # Create trainer
    trainer = GenomicTransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=training_config,
        class_names=class_names
    )
    
    # Override device in trainer
    trainer.device = device
    trainer.model = trainer.model.to(device)
    
    # Setup profiling if needed
    if mac_config.profiling:
        try:
            import torch.profiler
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    # MPS profiling not available yet, uncomment when available
                    # torch.profiler.ProfilerActivity.MPS
                ],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(mac_config.output_dir, "profiling")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            # Store profiler for manual activation in training loop
            trainer.profiler = profiler
        except ImportError:
            print("Profiling requested but torch.profiler not available - skipping")
            trainer.profiler = None
    else:
        trainer.profiler = None
    
    # Train the model
    history = trainer.train()
    
    # Return the model and history
    return model, history


# Example usage
if __name__ == "__main__":
    
    train_csv = "test_data/test_data.csv"
    # Create Mac testing config
    mac_config = MacTestingConfig(
        output_dir="mac_test_outputs",
        use_mps=True,
        memory_efficient=True,
        batch_size=2,
        test_epochs=2
    )

    model_config = get_small_model_config()
    
    # Create datasets
    train_dataset = SimplePathogenDataset(
        csv_file=train_csv,
        max_contigs=model_config.max_contigs,
        max_contig_len=model_config.max_seq_len,
        kmer_sizes=model_config.kmer_sizes,  # Use the SAME kmer_sizes as the model, or fail
        cache_dir="cache"
    )
    val_dataset = SimplePathogenDataset(
        csv_file=train_csv,
        max_contigs=model_config.max_contigs,
        max_contig_len=model_config.max_seq_len,
        kmer_sizes=model_config.kmer_sizes,  # Use the SAME kmer_sizes as the model, or Fail
        cache_dir="cache"
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=mac_config.batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=mac_config.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    # Run Mac testing
    model, history = run_mac_testing(
        mac_config=mac_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        class_names=train_dataset.species
    )
    
    print("Mac testing completed successfully!")