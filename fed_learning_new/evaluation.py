# -*- coding: utf-8 -*-
# Global model evaluation function. Python 3.7 compatible.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Import DataLoader type hint
import numpy as np # For accuracy calculation if needed, though direct sum is fine
import time
import os
import traceback

# Assumes setup_utils is available for logging
try:
    from .setup_utils import log_operation
    from .config import SERVER_ROLE_NAME
except ImportError:
    # Provide dummy logger and constant if run standalone
    def log_operation(log_dir, round_num, role_name, op_type, msg): pass
    SERVER_ROLE_NAME = "server"


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    log_dir: str,
    round_num: int,
    model_type: str
) -> float: # Returns the primary evaluation metric (Accuracy)
    """Evaluates the global model on the test dataset."""
    print(f"\n>> Evaluating Global Model (After Round {round_num}), Model: {model_type.upper()}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"   Evaluating on device: {device}")

    if not test_loader:
        print("[ERROR] Test loader is invalid or empty. Cannot evaluate.")
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "evaluate_fail", "Invalid test loader")
        return 0.0 # Return default metric value

    # Select loss and metric
    if model_type in ['mlp', 'cnn']:
        criterion = nn.CrossEntropyLoss()
        metric_name = "Accuracy"
        print("   Using Accuracy / CrossEntropyLoss for evaluation.")
    else:
        print(f"[ERROR] Unsupported model type for evaluation: {model_type}. Using Accuracy/CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()
        metric_name = "Accuracy"

    total_loss = 0.0
    correct = 0
    total_samples = 0
    final_metric_value = 0.0

    log_filename = os.path.join(log_dir, f"evaluation_log_round_{round_num}.txt")

    try:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        start_time = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() # Accumulate batch loss

                total_samples += target.size(0)

                if model_type in ['mlp', 'cnn']:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == target).sum().item()

        # Calculate final metrics
        if model_type in ['mlp', 'cnn']:
            accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
            final_metric_value = accuracy
            avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0 # Avg loss per batch
            log_info_metric = f'  {metric_name} (R{round_num}): {final_metric_value:.2f}% ({correct}/{total_samples})'
            log_info_loss = f'  Avg Loss (R{round_num}): {avg_loss:.4f}'

        end_time = time.time()
        eval_time = end_time - start_time

        # Log results to console and file
        print(log_info_metric)
        print(log_info_loss)
        print(f"   Evaluation Time: {eval_time:.2f}s")
        try:
             with open(log_filename, 'w') as log_file:
                 log_file.write(f"Evaluation Log - Round: {round_num}, Model: {model_type.upper()}\n")
                 log_file.write(f"Device: {device}\n")
                 log_file.write("-" * 30 + "\n")
                 log_file.write(log_info_metric + '\n')
                 log_file.write(log_info_loss + '\n')
                 log_file.write(f"Evaluation Time: {eval_time:.2f}s\n")
        except IOError as e:
             print(f"[ERROR] Could not write evaluation log to {log_filename}: {e}")
             log_operation(log_dir, round_num, SERVER_ROLE_NAME, "eval_log_fail", f"IOError: {e}")


        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "evaluate_finish", f"{metric_name}:{final_metric_value:.4f}, Time:{eval_time:.2f}s")

    except Exception as e:
        print(f"[ERROR] Evaluation failed in round {round_num}: {e}")
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "evaluate_exception", str(e))
        traceback.print_exc()
        return 0.0 # Return default value on error

    print(">> Evaluation Finished!")
    return final_metric_value