import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from dataclasses import dataclass, field
from model import GenomicTransformerV2


@dataclass
class TrainingConfig:
    """Configuration for training the Genomic Transformer Model."""
    # Training parameters
    output_dir: str = "outputs"
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False
    
    # Logging and checkpointing
    logging_steps: int = 100
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Scheduler
    scheduler_type: str = "cosine"  # Options: "linear", "cosine", "plateau"
    scheduler_patience: int = 2
    
    # Mixed precision
    use_mixed_precision: bool = False
    
    # Optimizer
    optimizer_type: str = "adamw"  # Options: "adamw", "adam", "sgd"
    
    # Class weights for imbalanced datasets
    use_class_weights: bool = False
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])


class GenomicTransformerTrainer:
    def __init__(
        self, 
        model: GenomicTransformerV2,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        test_dataloader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The GenomicTransformerV2 model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            config: Training configuration
            test_dataloader: Optional DataLoader for test data
            class_names: Optional list of class names for reporting
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.class_names = class_names
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Set the seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Set up output directory
        self.output_dir = Path(config.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({k: str(v) for k, v in vars(config).items()}, f, indent=4)
        
        # Set up device - in this case cpu will be 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.use_mixed_precision else None
        
        # Set up class weights if needed
        class_weights = None
        if config.use_class_weights:
            # Count samples per class from the train dataset
            labels = torch.cat([batch["label"] for batch in train_dataloader])
            class_counts = torch.bincount(labels)
            class_weights = torch.tensor(
                len(labels) / (len(class_counts) * class_counts), 
                dtype=torch.float32, 
                device=self.device
            )
            self.logger.info(f"Using class weights: {class_weights}")
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Set up optimizer
        self.optimizer = self._get_optimizer()
        
        # Set up scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize tracking variables
        self.best_val_metric = float('-inf')
        self.no_improvement_count = 0
        self.global_step = 0
        self.epoch = 0
        
        # Initialize metrics history
        self.train_metrics_history = []
        self.val_metrics_history = []
        
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer based on the config."""
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
            
    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler based on the config."""
        if self.config.scheduler_type.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        elif self.config.scheduler_type.lower() == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
            )
        elif self.config.scheduler_type.lower() == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=len(self.train_dataloader) * self.config.num_epochs
            )
        else:
            self.logger.info(f"No scheduler used (type: {self.config.scheduler_type})")
            return None
            
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dict containing training history
        """
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Number of training examples: {len(self.train_dataloader.dataset)}")
        self.logger.info(f"Number of validation examples: {len(self.val_dataloader.dataset)}")
        if self.test_dataloader:
            self.logger.info(f"Number of test examples: {len(self.test_dataloader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            self.train_metrics_history.append(train_metrics)
            
            # Validation phase
            val_metrics = self._validate_epoch()
            self.val_metrics_history.append(val_metrics)
            
            # Update learning rate if using ReduceLROnPlateau
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["accuracy"])
            
            # Save checkpoint
            self._save_checkpoint(
                filename=f"checkpoint_epoch_{epoch+1}.pt", 
                metrics=val_metrics
            )
            
            # Early stopping check
            if val_metrics["accuracy"] > self.best_val_metric + self.config.early_stopping_threshold:
                self.best_val_metric = val_metrics["accuracy"]
                self.no_improvement_count = 0
                # Save best model
                self._save_checkpoint(
                    filename="best_model.pt",
                    metrics=val_metrics
                )
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Load the best model and evaluate on test set if available
        if self.test_dataloader:
            self.logger.info("Evaluating best model on test set")
            best_model_path = self.output_dir / "best_model.pt"
            if best_model_path.exists():
                self._load_checkpoint(best_model_path)
                test_metrics = self._evaluate(self.test_dataloader, prefix="test")
                
                # Save test metrics
                with open(self.output_dir / "test_metrics.json", "w") as f:
                    json.dump(test_metrics, f, indent=4)
                
                self.logger.info(f"Test metrics: {test_metrics}")
            else:
                self.logger.warning("Best model checkpoint not found. Skipping test evaluation.")
        
        # Save training history
        history = {
            "train": self.train_metrics_history,
            "validation": self.val_metrics_history
        }
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        return history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar for the training epoch
        progress_bar = tqdm(
            enumerate(self.train_dataloader), 
            total=len(self.train_dataloader),
            desc=f"Epoch {self.epoch+1} [Train]"
        )
        
        for step, batch in progress_bar:
            # Move batch to device
            kmer_indices = [indices.to(self.device) for indices in batch["kmer_indices"]]
            padding_mask = batch["padding_mask"].to(self.device)
            assembly_features = batch["assembly_features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(kmer_indices, padding_mask, assembly_features)
                    loss = self.criterion(outputs, labels)
                
                # Scale the loss and backpropagation
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Forward pass
                outputs = self.model(kmer_indices, padding_mask, assembly_features)
                loss = self.criterion(outputs, labels)
                
                # Backpropagation
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update learning rate if using non-plateau schedulers
            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / (step + 1)
            })
            
            # Log metrics at specified intervals
            self.global_step += 1
            if self.global_step % self.config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss = {loss.item():.4f}, "
                    f"lr = {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Save model at specified intervals
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(
                    filename=f"checkpoint_step_{self.global_step}.pt",
                    metrics={"loss": loss.item()}
                )
        
        # Compute metrics
        train_metrics = self._compute_metrics(all_labels, all_preds)
        train_metrics["loss"] = total_loss / len(self.train_dataloader)
        
        # Log metrics
        self.logger.info(f"Epoch {self.epoch+1} training metrics: {train_metrics}")
        
        return train_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate the model on the validation set."""
        return self._evaluate(self.val_dataloader, prefix="val")
    
    def _evaluate(self, dataloader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """
        Evaluate the model on a given dataloader.
        
        Args:
            dataloader: DataLoader containing the evaluation data
            prefix: String prefix for logging (e.g., "val", "test")
            
        Returns:
            Dict of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar for evaluation
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader),
            desc=f"Epoch {self.epoch+1} [{prefix.capitalize()}]"
        )
        
        with torch.no_grad():
            for step, batch in progress_bar:
                # Move batch to device
                kmer_indices = [indices.to(self.device) for indices in batch["kmer_indices"]]
                padding_mask = batch["padding_mask"].to(self.device)
                assembly_features = batch["assembly_features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(kmer_indices, padding_mask, assembly_features)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "avg_loss": total_loss / (step + 1)
                })
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(dataloader)
        
        # Generate classification report with safety checks
        if self.class_names:
            # Get unique classes in predictions/labels
            unique_classes = sorted(set(all_labels).union(set(all_preds)))
            num_unique_classes = len(unique_classes)
            
            # Check if class_names matches number of unique classes
            if len(self.class_names) != num_unique_classes:
                self.logger.warning(
                    f"Number of classes in data ({num_unique_classes}) does not match "
                    f"number of class names ({len(self.class_names)}). "
                    f"Using generic class names instead."
                )
                # Use generic class names
                target_names = [f"Class {i}" for i in range(num_unique_classes)]
            else:
                target_names = self.class_names
            
            # Generate and save report
            report = classification_report(
                all_labels, all_preds, 
                target_names=target_names,
                output_dict=True
            )
            
            # Save detailed classification report
            report_path = self.output_dir / f"{prefix}_classification_report_epoch_{self.epoch+1}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
        
        # Log metrics
        self.logger.info(f"Epoch {self.epoch+1} {prefix} metrics: {metrics}")
        
        return metrics
    def _compute_metrics(self, labels: List[int], preds: List[int]) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            labels: List of true labels
            preds: List of predicted labels
            
        Returns:
            Dict of computed metrics
        """
        accuracy = accuracy_score(labels, preds)
        
        # Calculate class-wise metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        return metrics
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """
        Save a checkpoint of the model and training state.
        
        Args:
            filename: Name of the checkpoint file
            metrics: Dictionary of metrics to save with the checkpoint
        """
        checkpoint_path = self.output_dir / filename
        
        # Limit the number of saved checkpoints
        if self.config.save_total_limit > 0:
            checkpoints = sorted(
                [f for f in self.output_dir.glob("checkpoint_*.pt")],
                key=lambda f: f.stat().st_mtime
            )
            if len(checkpoints) >= self.config.save_total_limit:
                # Remove oldest checkpoints
                for old_ckpt in checkpoints[:len(checkpoints) - self.config.save_total_limit + 1]:
                    old_ckpt.unlink()
        
        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "best_val_metric": self.best_val_metric
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict() if not isinstance(self.scheduler, ReduceLROnPlateau)
                else {"best": self.scheduler.best, "mode": self.scheduler.mode}
            )
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if training
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                # Handle ReduceLROnPlateau separately as it doesn't have a proper state_dict
                self.scheduler.best = checkpoint["scheduler_state_dict"]["best"]
                self.scheduler.mode = checkpoint["scheduler_state_dict"]["mode"]
        
        # Load training state
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        if "best_val_metric" in checkpoint:
            self.best_val_metric = checkpoint["best_val_metric"]
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
