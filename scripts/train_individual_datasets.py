#!/usr/bin/env python3
"""
Individual Dataset Training for MARU with Enhanced MoM-GRU

This script trains MARU on individual datasets sequentially, allowing for:
- Focused analysis of performance on each dataset type
- Better understanding of memory specialization
- Easier debugging and monitoring
- Checkpoint management per dataset

MSC RERUN NOTES:
- MSC was the first dataset trained with enhanced MoM-GRU architecture
- Initial run showed poor performance likely due to early implementation issues
- This version includes optimized MSC processing with enhanced memory context
- Adds previous dialog context for better memory persistence testing
- Timestamps MSC models to distinguish from original baseline run
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import psutil

# Add src to path for imports
sys.path.append('src')

from maru import MARU, MARUConfig
from tokenizer import CharacterTokenizer
from enhanced_mom_gru_config import get_conservative_config, get_full_config



# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/individual_training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dataset configurations - OPTIMIZED FOR 12GB VRAM WITH MAXIMUM PARAMETERS
# Key insight: Activation memory scales with batch_size × sequence_length
# Parameter memory is fixed. Reduce activations to maximize parameters.
DATASET_CONFIGS = {
    'lambada': {
        'file': 'data/external_datasets/lambada_formatted.jsonl',
        'name': 'LAMBADA (Long-range Dependencies)',
        'description': 'Tests ability to predict words requiring long-range context',
        'max_epochs': 3,
        'batch_size': 2,  # Very conservative for 12GB VRAM
        'learning_rate': 1e-4
    },
    'msc': {
        'file': 'data/external_datasets/msc_formatted.jsonl',
        'name': 'MSC (Multi-Session Memory) - RERUN',
        'description': 'Tests persistent memory across conversation sessions - Re-running with optimized architecture',
        'max_epochs': 5,
        'batch_size': 2,  # Very conservative for 12GB VRAM
        'learning_rate': 8e-5
    },
    'narrativeqa': {
        'file': 'data/external_datasets/narrativeqa_formatted.jsonl',
        'name': 'NarrativeQA (Long Document QA)',
        'description': 'Tests comprehension of long documents and question answering',
        'max_epochs': 4,
        'batch_size': 2,  # Very conservative for 12GB VRAM
        'learning_rate': 6e-5
    },
    'mixed': {
        'file': 'data/external_datasets/maru_external_mixed_dataset.jsonl',
        'name': 'Mixed Dataset',
        'description': 'Combined dataset testing all capabilities',
        'max_epochs': 6,
        'batch_size': 2,  # Very conservative for 12GB VRAM
        'learning_rate': 1e-4
    }
}

class IndividualDataset:
    """Dataset class for individual dataset training."""
    
    def __init__(self, data_file: str, tokenizer: CharacterTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {data_file}...")
        
        if not os.path.exists(data_file):
            logger.error(f"Dataset file not found: {data_file}")
            return
        
        # Count lines for progress bar
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            logger.info(f"Found {total_lines} lines to process")
        except Exception as e:
            logger.error(f"Error counting lines in {data_file}: {e}")
            return

        # Load with progress bar
        processed_count = 0
        error_count = 0

        try:
            with tqdm(total=total_lines, desc="Loading dataset", unit="lines",
                     disable=False, dynamic_ncols=True) as pbar:
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                pbar.update(1)
                                continue

                            data = json.loads(line)
                            processed = self._process_example(data)
                            if processed:
                                self.examples.append(processed)
                                processed_count += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error on line {line_num}: {e}")
                            error_count += 1
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num}: {e}")
                            error_count += 1
                        finally:
                            pbar.update(1)

                            # Update progress description periodically
                            if line_num % 100 == 0:
                                pbar.set_description(f"Loading dataset (processed: {processed_count}, errors: {error_count})")

        except Exception as e:
            logger.error(f"Fatal error loading dataset: {e}")
            return
        
        logger.info(f"Loaded {len(self.examples)} training examples (processed: {processed_count}, errors: {error_count})")

        if len(self.examples) == 0:
            logger.error("No valid examples loaded from dataset!")
            return
    
    def _process_example(self, data: Dict) -> Optional[Dict]:
        """Process different dataset formats into unified training format."""
        task_type = data.get('task_type', 'unknown')
        
        if task_type == 'multi_session_memory':
            return self._process_msc(data)
        elif task_type == 'long_range_prediction':
            return self._process_lambada(data)
        elif task_type == 'long_document_qa':
            return self._process_qa(data)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return None
    
    def _process_msc(self, data: Dict) -> Optional[Dict]:
        """Process MSC dataset format with enhanced memory context."""
        try:
            # MSC data has dialog nested under session_data
            session_data = data.get('session_data', {})
            dialog = session_data.get('dialog', [])

            # If no dialog in session_data, try top level (fallback)
            if not dialog:
                dialog = data.get('dialog', [])

            if not dialog:
                return None

            # Enhanced MSC processing: Include previous dialog context for memory testing
            previous_dialogs = session_data.get('previous_dialogs', [])

            # Build context from previous sessions (for memory persistence testing)
            context_parts = []

            # Add previous dialog summaries if available
            if previous_dialogs:
                for prev_dialog in previous_dialogs[-2:]:  # Last 2 previous sessions
                    prev_turns = prev_dialog.get('dialog', [])
                    if prev_turns:
                        # Summarize previous session
                        prev_text = ' '.join([turn.get('text', '') for turn in prev_turns[-3:]])  # Last 3 turns
                        context_parts.append(f"[PREV] {prev_text}")

            # Add current dialog
            current_text = ' '.join([turn.get('text', '') for turn in dialog])
            context_parts.append(f"[CURR] {current_text}")

            # Combine all context
            full_text = ' '.join(context_parts)

            # Tokenize
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) > self.max_length:
                # Prioritize current dialog if we need to truncate
                current_tokens = self.tokenizer.encode(f"[CURR] {current_text}")
                if len(current_tokens) <= self.max_length:
                    tokens = current_tokens
                else:
                    tokens = tokens[:self.max_length]

            # Create input and target sequences
            input_ids = tokens[:-1] if len(tokens) > 1 else tokens
            target_ids = tokens[1:] if len(tokens) > 1 else tokens

            return {
                'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
                'target_ids': torch.as_tensor(target_ids, dtype=torch.long),
                'task_type': 'multi_session_memory'
            }
        except Exception as e:
            logger.warning(f"Error processing MSC example: {e}")
            return None
    
    def _process_lambada(self, data: Dict) -> Optional[Dict]:
        """Process LAMBADA dataset format."""
        try:
            context = data.get('context_without_target', '')
            target_word = data.get('target_word', '')
            
            if not context or not target_word:
                return None
            
            # Combine context and target
            full_text = context + ' ' + target_word
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Create input and target sequences
            input_ids = tokens[:-1] if len(tokens) > 1 else tokens
            target_ids = tokens[1:] if len(tokens) > 1 else tokens
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'task_type': 'long_range_prediction'
            }
        except Exception as e:
            logger.warning(f"Error processing LAMBADA example: {e}")
            return None
    
    def _process_qa(self, data: Dict) -> Optional[Dict]:
        """Process QA dataset format."""
        try:
            # Handle different field names for different QA datasets
            context = data.get('context', '') or data.get('document', '')
            question = data.get('question', '')

            # Handle both single answer and answer arrays
            answer = data.get('answer', '')
            if not answer:
                answers = data.get('answers', [])
                if answers:
                    answer = answers[0] if isinstance(answers, list) else str(answers)

            if not all([context, question, answer]):
                return None

            # Format as: context + question + answer
            full_text = f"{context} Q: {question} A: {answer}"
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Create input and target sequences
            input_ids = tokens[:-1] if len(tokens) > 1 else tokens
            target_ids = tokens[1:] if len(tokens) > 1 else tokens
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'task_type': 'long_document_qa'
            }
        except Exception as e:
            logger.warning(f"Error processing QA example: {e}")
            return None
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Optimized collate function for DataLoader."""
    # Pre-allocate lists for better performance
    batch_size = len(batch)
    max_len = max(len(item['input_ids']) for item in batch)

    # Pre-allocate tensors with padding
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    target_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        # Get sequence lengths
        input_seq = item['input_ids']
        target_seq = item['target_ids']

        # Convert to tensor if needed
        if not torch.is_tensor(input_seq):
            input_seq = torch.tensor(input_seq, dtype=torch.long)
        if not torch.is_tensor(target_seq):
            target_seq = torch.tensor(target_seq, dtype=torch.long)

        # Copy into pre-allocated tensors (automatically pads with zeros)
        input_len = min(len(input_seq), max_len)
        target_len = min(len(target_seq), max_len)

        input_ids[i, :input_len] = input_seq[:input_len]
        target_ids[i, :target_len] = target_seq[:target_len]

    return {
        'input_ids': input_ids,
        'target_ids': target_ids
    }


def get_memory_usage():
    """Get current GPU VRAM usage in MB."""
    if torch.cuda.is_available():
        # Get actual allocated memory (more accurate)
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # Fallback to system RAM if no CUDA
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

def get_detailed_memory_info():
    """Get detailed GPU memory information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return f"Allocated: {allocated:.0f}MB, Reserved: {reserved:.0f}MB, Total: {total:.0f}MB"
    return "CUDA not available"


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, dataset_name, config, scaler=None, batch_offset=0):
    """Train for one epoch with progress tracking and mixed precision support."""
    # Aggressive memory cleanup before starting epoch
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Set memory fraction to prevent fragmentation (more conservative)
        torch.cuda.set_per_process_memory_fraction(0.75)
        # Enable memory pool for better allocation with smaller splits
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    use_amp = scaler is not None

    print(f"\n=== {dataset_name} - Epoch {epoch}/{total_epochs} ===")
    print(f"Batches: {num_batches}, Device: {device}")

    # Debug: Log before starting training loop
    logger.info(f"About to start training loop with {num_batches} batches")
    if batch_offset > 0:
        logger.info(f"Resuming from batch {batch_offset}, skipping {batch_offset} batches")
    logger.info(f"Current VRAM before training: {get_memory_usage():.0f}MB")

    # Create progress bar for batches
    pbar = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc=f"{dataset_name} Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
        initial=batch_offset  # Start progress bar from the resume point
    )

    # Debug: Log after creating progress bar
    logger.info("Progress bar created, starting batch iteration...")

    epoch_start_time = time.time()
    processed_batches = 0  # Track actual number of batches processed

    for batch_idx, batch in pbar:
        # Skip batches if resuming from checkpoint
        if batch_idx < batch_offset:
            continue

        batch_start_time = time.time()
        processed_batches += 1  # Increment for each batch we actually process

        # Debug: Log batch loading
        if batch_idx == batch_offset:  # First batch we actually process
            logger.info(f"Processing batch {batch_idx} (resuming from batch {batch_offset}). Batch size: {len(batch['input_ids'])}")
            logger.info(f"Input shape: {batch['input_ids'].shape}, Target shape: {batch['target_ids'].shape}")

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # Debug: Log device transfer
        if batch_idx == 0:
            logger.info(f"Batch moved to device: {device}")
            logger.info(f"Memory info: {get_detailed_memory_info()}")
            logger.info("About to start forward pass...")

        # Forward pass with mixed precision support
        optimizer.zero_grad()

        # Debug: Log after zero_grad
        if batch_idx == 0:
            logger.info("Gradients zeroed, starting model forward pass...")

        if use_amp:
            with autocast():
                # Debug: Log before model call
                if batch_idx == 0:
                    logger.info("Calling model forward pass with mixed precision...")

                # Get model output
                output, _ = model(input_ids, return_hidden=False)

                # Debug: Log after model call
                if batch_idx == 0:
                    logger.info(f"Model forward pass completed. Output shape: {output.shape}")

                # Compute main loss
                batch_size, seq_len, vocab_size = output.shape
                output_flat = output.view(-1, vocab_size)
                target_flat = target_ids.view(-1)

                main_loss = criterion(output_flat, target_flat)

            # Backward pass with gradient scaling
            scaler.scale(main_loss).backward()

            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients after step (additional safety)
            optimizer.zero_grad()
        else:
            # Standard precision training
            output, _ = model(input_ids, return_hidden=False)

            # Compute main loss
            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.view(-1, vocab_size)
            target_flat = target_ids.view(-1)

            main_loss = criterion(output_flat, target_flat)

            # Backward pass
            main_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Clear gradients after step (additional safety)
        optimizer.zero_grad()

        # Update running averages (detach to prevent graph accumulation)
        total_loss += main_loss.detach().item()

        # Calculate current averages (use processed_batches, not batch_idx)
        current_avg_loss = total_loss / processed_batches

        batch_time = time.time() - batch_start_time

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'mem': f'{get_memory_usage():.0f}MB',
            'time': f'{batch_time:.2f}s'
        })
        # Update progress bar position to show actual batch number
        pbar.n = batch_idx + 1
        pbar.refresh()

        # Log detailed info
        logger.info(f"{dataset_name} - Epoch {epoch}/{total_epochs}, Batch {batch_idx+1}/{num_batches}, "
                   f"Loss: {main_loss.item():.6f}, "
                   f"Avg Loss: {current_avg_loss:.6f}, Memory: {get_memory_usage():.0f}MB, "
                   f"Batch Time: {batch_time:.2f}s")

        # Periodic memory cleanup to prevent leaks
        if batch_idx > 0 and batch_idx % 5 == 0:
            torch.cuda.empty_cache()  # Clear GPU cache

        # More aggressive cleanup every 2 batches due to memory fragmentation issues
        if batch_idx > 0 and batch_idx % 2 == 0:
            # Force garbage collection
            import gc
            gc.collect()
            # Clear GPU cache more aggressively
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Clear any cached tensors
            if hasattr(model, 'clear_memory_cache'):
                model.clear_memory_cache()

        # Save mid-epoch checkpoint every 50 batches
        if batch_idx > 0 and batch_idx % 50 == 0:
            mid_checkpoint_path = f"checkpoints/{dataset_name}_epoch_{epoch}_batch_{batch_idx}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            print(f"  >> Saving mid-epoch checkpoint: {mid_checkpoint_path}")
            logger.info(f"Saving mid-epoch checkpoint: {mid_checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'dataset': dataset_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'main_loss': current_avg_loss,
                'config': config,
                'is_mid_epoch': True
            }, mid_checkpoint_path)

    pbar.close()
    epoch_time = time.time() - epoch_start_time

    avg_main_loss = total_loss / processed_batches

    print(f"  >> Epoch completed in {epoch_time:.1f}s")
    print(f"  >> Average Loss: {avg_main_loss:.6f}")

    return avg_main_loss


def train_on_dataset(dataset_name: str, config: Dict, model: MARU, tokenizer: CharacterTokenizer,
                    optimizer: AdamW, criterion: nn.Module, device: torch.device, scaler=None, resume_info=None) -> float:
    """Train model on a specific dataset."""

    dataset_config = DATASET_CONFIGS[dataset_name]

    print(f"\n{'='*80}")
    print(f"TRAINING ON: {dataset_config['name']}")
    print(f"Description: {dataset_config['description']}")
    print(f"File: {dataset_config['file']}")
    print(f"Epochs: {dataset_config['max_epochs']}")
    print(f"Batch Size: {dataset_config['batch_size']}")
    print(f"Learning Rate: {dataset_config['learning_rate']}")
    print(f"{'='*80}")

    # Update optimizer learning rate for this dataset
    for param_group in optimizer.param_groups:
        param_group['lr'] = dataset_config['learning_rate']

    # Create dataset and dataloader
    dataset = IndividualDataset(dataset_config['file'], tokenizer, config['max_length'])

    if len(dataset) == 0:
        logger.error(f"No examples loaded for {dataset_name}")
        return float('inf')

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    logger.info(f"Starting training on {dataset_name} with {len(dataset)} examples")

    best_loss = float('inf')

    # Determine starting epoch and batch for resumption
    start_epoch = 1
    start_batch = 0
    if resume_info:
        start_epoch = resume_info['epoch']
        start_batch = resume_info['batch'] + 1  # Resume from next batch
        print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
        logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")

    # Training loop for this dataset
    for epoch in range(start_epoch, dataset_config['max_epochs'] + 1):
        epoch_start = time.time()

        # Train epoch (with potential batch offset for resumption)
        batch_offset = start_batch if epoch == start_epoch else 0
        avg_loss = train_epoch(
            model, dataloader, optimizer, criterion,
            device, epoch, dataset_config['max_epochs'], dataset_name, config, scaler, batch_offset
        )

        epoch_time = time.time() - epoch_start

        # Track best performance
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Save checkpoint after each epoch
        checkpoint_path = f"checkpoints/{dataset_name}_epoch_{epoch}_final.pt"
        os.makedirs("checkpoints", exist_ok=True)
        print(f">> Saving epoch checkpoint: {checkpoint_path}")
        logger.info(f"Saving epoch checkpoint: {checkpoint_path}")
        torch.save({
            'epoch': epoch,
            'dataset': dataset_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'main_loss': avg_loss,
            'best_loss': best_loss,
            'config': config,
            'is_final_epoch': epoch == dataset_config['max_epochs']
        }, checkpoint_path)

        print(f">> Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f">> Loss: {avg_loss:.6f} (Best: {best_loss:.6f})")

        # Get monitoring stats if available
        monitoring_stats = model.get_monitoring_stats()
        if monitoring_stats:
            logger.info(f"Monitoring stats for {dataset_name} epoch {epoch}: {monitoring_stats}")

    print(f"\n>> COMPLETED TRAINING ON {dataset_config['name']}")
    print(f">> Best Loss: {best_loss:.6f}")

    return best_loss



def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MARU on individual datasets')
    parser.add_argument('--datasets', nargs='+',
                       choices=['lambada', 'msc', 'narrativeqa', 'mixed', 'all'],
                       default=['msc'],
                       help='Datasets to train on (default: msc for rerun)')
    parser.add_argument('--enhanced-config', choices=['conservative', 'full', 'baseline'],
                       default='conservative',
                       help='Enhanced MoM-GRU configuration (default: conservative)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length (default: 512, conservative for 12GB VRAM)')
    parser.add_argument('--vocab-size', type=int, default=256,
                       help='Vocabulary size (default: 256)')
    parser.add_argument('--d-model', type=int, default=640,
                       help='Model dimension (default: 640, conservative for 12GB VRAM)')
    parser.add_argument('--hidden-size', type=int, default=640,
                       help='Hidden size (default: 640, conservative for 12GB VRAM)')
    parser.add_argument('--memory-size', type=int, default=256,
                       help='Memory size (default: 256, conservative for 12GB VRAM)')
    parser.add_argument('--memory-dim', type=int, default=128,
                       help='Memory dimension (default: 128, conservative for 12GB VRAM)')
    parser.add_argument('--num-memories', type=int, default=6,
                       help='Number of memory banks (default: 6, conservative for 12GB VRAM)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda) (default: auto)')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training (AMP) - default: True for memory savings')
    parser.add_argument('--compile-model', action='store_true',
                       help='Compile model with torch.compile for optimization')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                       help='Use gradient checkpointing to save memory (default: True)')
    parser.add_argument('--sequence-filter', type=int, default=None,
                       help='Filter sequences longer than this length')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='Number of gradient accumulation steps (default: 4 to simulate larger batches)')



    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Determine datasets to train on - FORCE MSC FOR RERUN
    # Override any command line arguments to ensure MSC rerun
    datasets_to_train = ['msc']
    print("🔄 FORCED MSC RERUN MODE: Ignoring any other dataset arguments")
    logger.info("Forced MSC rerun mode: Training only MSC dataset")

    # Special handling for MSC rerun - add timestamp to distinguish from previous run
    if 'msc' in datasets_to_train:
        print("🔄 MSC RERUN MODE: Training MSC with optimized enhanced MoM-GRU architecture")
        logger.info("MSC rerun mode: Using optimized architecture after initial baseline issues")

    print(f"Training on datasets: {datasets_to_train}")
    logger.info(f"Training on datasets: {datasets_to_train}")

    # Create configuration
    config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'hidden_size': args.hidden_size,
        'max_length': args.max_length,
        'memory_size': args.memory_size,
        'memory_dim': args.memory_dim,
        'num_memories': args.num_memories,
        'device': device
    }

    # Initialize tokenizer
    tokenizer = CharacterTokenizer()

    # Create Enhanced MoM-GRU configuration
    if args.enhanced_config == 'conservative':
        enhanced_config = get_conservative_config()
    elif args.enhanced_config == 'full':
        enhanced_config = get_full_config()
    else:  # baseline
        enhanced_config = None

    # Initialize model
    maru_config = MARUConfig(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        hidden_size=config['hidden_size'],
        output_dim=config['vocab_size'],
        memory_size=config['memory_size'],
        memory_dim=config['memory_dim'],
        num_memories=config['num_memories'],
        use_enhanced_mom_gru=(enhanced_config is not None),
        enhanced_mom_gru_config=enhanced_config,
        device=device
    )

    model = MARU(maru_config)
    model = model.to(device)

    # Compile model if requested
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Enhanced MoM-GRU: {maru_config.use_enhanced_mom_gru}")
    logger.info(f"Mixed Precision: {args.mixed_precision}")
    logger.info(f"Model Compilation: {args.compile_model}")

    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=1e-4)  # Will be adjusted per dataset
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    # Initialize mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        logger.info("Mixed precision training enabled with GradScaler")

    print("📚 Standard training mode")
    logger.info("Standard training mode")

    # Resume from checkpoint if specified
    start_dataset_idx = 0
    resume_info = None
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Check if this is a mid-epoch checkpoint
            if checkpoint.get('is_mid_epoch', False):
                resume_info = {
                    'dataset': checkpoint['dataset'],
                    'epoch': checkpoint['epoch'],
                    'batch': checkpoint['batch']
                }
                print(f"Resuming mid-epoch: {resume_info['dataset']} epoch {resume_info['epoch']} from batch {resume_info['batch'] + 1}")
                logger.info(f"Mid-epoch resume: {resume_info['dataset']} epoch {resume_info['epoch']} batch {resume_info['batch'] + 1}")
            else:
                # Determine which dataset to start from (end-of-epoch checkpoint)
                completed_dataset = checkpoint.get('dataset', '')
                if completed_dataset in datasets_to_train:
                    start_dataset_idx = datasets_to_train.index(completed_dataset) + 1
                    print(f"Resuming after {completed_dataset}, starting from dataset index {start_dataset_idx}")
        else:
            logger.warning(f"Checkpoint file not found: {args.resume_from}")

    # Training loop
    print(f"\n{'='*100}")
    print(f"STARTING INDIVIDUAL DATASET TRAINING")
    print(f"Enhanced MoM-GRU Config: {args.enhanced_config}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Datasets: {datasets_to_train}")
    print(f"{'='*100}")

    start_time = time.time()
    results = {}

    # Train on each dataset sequentially
    for dataset_idx, dataset_name in enumerate(datasets_to_train[start_dataset_idx:], start_dataset_idx):
        dataset_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"DATASET {dataset_idx + 1}/{len(datasets_to_train)}: {dataset_name.upper()}")
        print(f"{'='*60}")

        try:
            # Check if we need to resume this specific dataset
            dataset_resume_info = None
            if resume_info and resume_info['dataset'] == dataset_name:
                dataset_resume_info = resume_info

            best_loss = train_on_dataset(
                dataset_name, config, model, tokenizer, optimizer, criterion, device, scaler, dataset_resume_info
            )

            results[dataset_name] = {
                'best_loss': best_loss,
                'training_time': time.time() - dataset_start_time
            }

            # Save final model for this dataset with timestamp for MSC rerun
            if dataset_name == 'msc':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_model_path = f"models/{dataset_name}_rerun_{timestamp}_final_model.pt"
            else:
                final_model_path = f"models/{dataset_name}_final_model.pt"

            os.makedirs("models", exist_ok=True)
            logger.info(f"Saving final model for {dataset_name}: {final_model_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'maru_config': maru_config,
                'tokenizer': tokenizer,
                'dataset': dataset_name,
                'results': results[dataset_name],
                'training_timestamp': datetime.now().isoformat(),
                'is_rerun': dataset_name == 'msc'
            }, final_model_path)

        except Exception as e:
            logger.error(f"Error training on {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}

    total_time = time.time() - start_time

    # Print final results
    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETED")
    print(f"Total Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"{'='*100}")

    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"{dataset_name:15}: ERROR - {result['error']}")
        else:
            print(f"{dataset_name:15}: Loss={result['best_loss']:.6f}, "
                  f"Time={result['training_time']:.1f}s")

    logger.info(f"Training completed. Results: {results}")


if __name__ == "__main__":
    main()
