# -*- coding: utf-8 -*-
# General utility functions.

import os
import datetime
import shutil

# ========== Logging Utility ==========
def log_operation(log_dir, round_num, role_name, operation_type, message):
    """Logs an operation to a central log file."""
    log_filename = os.path.join(log_dir, f"operations_log.txt")
    try:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_info = (
            f'{timestamp} | Round: {round_num:03d} | Role: {role_name:<12} | '
            f'Operation: {operation_type:<25} | Message: {message}'
        )
        print(f"[LOG] {log_info}") # Also print to console for immediate feedback
        with open(log_filename, 'a') as log_file:
            log_file.write(log_info + '\n')
    except Exception as e:
        print(f"[ERROR] Failed to write log to {log_filename}: {e}")

# ========== Directory Setup Utility ==========
def setup_logging_directory(log_dir):
    """Cleans up (if exists) and creates the logging directory."""
    try:
        if os.path.exists(log_dir):
            print(f"\n>> Clearing existing log directory: {log_dir}")
            # Use ignore_errors=True for robustness, e.g., if files are locked
            shutil.rmtree(log_dir, ignore_errors=True)
            # Add a small delay if needed, e.g., on Windows
            # import time; time.sleep(0.1)
        os.makedirs(log_dir, exist_ok=True)
        print(f">> Log directory created/cleaned: {log_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create log directory {log_dir}: {e}")
        return False