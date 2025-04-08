# -*- coding: utf-8 -*-
# Local model training function. Python 3.7 compatible.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Import DataLoader type hint
import time
import os
import traceback

# Assumes setup_utils is available for logging
try:
    from .setup_utils import log_operation
except ImportError:
    # Provide a dummy logger if run standalone
    def log_operation(log_dir, round_num, role_name, op_type, msg): pass

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    log_dir: str,
    round_num: int,
    participant_id: str,
    model_type: str
) -> nn.Module: # Return type is nn.Module
    """Performs local training for one participant."""
    print(f"\n>> Starting Local Training: {participant_id}, Round: {round_num}, Model: {model_type.upper()}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Training on device: {device}")

    if not train_loader or len(train_loader.dataset) == 0:
        print("[Warning] Training loader is empty. Skipping training.")
        log_operation(log_dir, round_num, participant_id, "train_skip", "Empty dataloader")
        return model # Return the original model

    # Select loss function
    if model_type in ['mlp', 'cnn']:
        criterion = nn.CrossEntropyLoss()
        print("   Using CrossEntropyLoss for training.")
    else:
         print(f"[ERROR] Unsupported model type for training loss: {model_type}. Using CrossEntropyLoss as fallback.")
         criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    log_filename = os.path.join(log_dir, f"train_log_round_{round_num}_{participant_id}.txt")

    model.train()
    total_loss = 0.0
    total_batches = 0

    try:
        # Ensure log directory exists before opening file
        os.makedirs(log_dir, exist_ok=True)
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Training Log - Participant: {participant_id}, Round: {round_num}, Model: {model_type.upper()}, Epochs: {epochs}\n")
            log_file.write(f"Device: {device}, Optimizer: AdamW, LR: {learning_rate}\n")
            log_file.write(f"Dataset size: {len(train_loader.dataset)}\n")
            log_file.write("-" * 30 + "\n")
            print(f"   Logging training details to: {log_filename}")

            start_time = time.time()
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                print(f"  Epoch {epoch+1}/{epochs}")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    current_loss = loss.item()
                    epoch_loss += current_loss
                    total_loss += current_loss
                    epoch_batches += 1
                    total_batches += 1

                    if batch_idx % 100 == 0:
                        batch_log = f'  Ep:{epoch+1}, B:{batch_idx}/{len(train_loader)}, Loss:{current_loss:.4f}'
                        print(batch_log)
                        log_file.write(batch_log + '\n')

                # Avoid division by zero if epoch has no batches
                avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
                epoch_summary = f"  Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}"
                print(epoch_summary)
                log_file.write(epoch_summary + '\n')

            end_time = time.time()
            training_time = end_time - start_time
            # Avoid division by zero if no batches were processed at all
            avg_total_loss = total_loss / total_batches if total_batches > 0 else 0
            final_log = (
                f">> Training Finished: {participant_id}, R:{round_num}! "
                f"Total Avg Loss: {avg_total_loss:.4f}, Time: {training_time:.2f}s"
            )
            print(final_log)
            log_file.write("-" * 30 + "\n")
            log_file.write(final_log + '\n')
            log_operation(log_dir, round_num, participant_id, "train_finish", f"AvgLoss:{avg_total_loss:.4f}, Time:{training_time:.2f}s")

    except FileNotFoundError:
         print(f"[ERROR] Could not write training log. Directory may not exist or permission denied: {log_dir}")
         log_operation(log_dir, round_num, participant_id, "train_log_fail", "FileNotFoundError")
    except Exception as e:
        print(f"[ERROR] Training failed for {participant_id} in round {round_num}: {e}")
        log_operation(log_dir, round_num, participant_id, "train_exception", str(e))
        traceback.print_exc()

    return model # Always return the model, even if training failed