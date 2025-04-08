# -*- coding: utf-8 -*-
# Model aggregation logic (FedAvg). Python 3.7 compatible.

import torch
import torch.nn as nn
import io
import base64
from typing import Union, List, Dict, Any # Use typing.Union

# Avoid direct model imports if possible, rely on serialization/deserialization
# Import the specific function needed
try:
    from .serialization import deserialize_model, _create_new_model_instance # Use helper
    from .setup_utils import log_operation # Assumes logger is in setup_utils
except ImportError:
    # Fallback for standalone testing
    def deserialize_model(model_str, model_type, dataset_name): return None
    def _create_new_model_instance(model_type, dataset_name): return None
    def log_operation(log_dir, round_num, role_name, op_type, msg): pass


def aggregate_global_model(
    base_model: nn.Module, # Pass the actual base model object
    participant_updates_list: List[Dict[str, str]],
    model_type: str,
    dataset_name: str,
    log_dir: str, # Pass log_dir for logging
    round_num: int  # Pass round_num for logging
) -> Union[nn.Module, None]: # Use Union
    """Aggregates participant model updates using FedAvg."""
    if not participant_updates_list or not isinstance(participant_updates_list, list):
        print("[Warning] Participant updates list empty/invalid. Returning base model.")
        return base_model

    print(f"\n>> Starting aggregation for {len(participant_updates_list)} participant updates...")

    try:
        # Create a new model instance for the aggregation result
        # Use the helper from serialization module to ensure consistency
        aggregated_model = _create_new_model_instance(model_type, dataset_name)
        if aggregated_model is None: raise ValueError("Failed to create new model instance.")
        aggregated_state_dict = aggregated_model.state_dict()
    except ValueError as e:
        print(f"[ERROR] Cannot create aggregated model instance: {e}")
        log_operation(log_dir, round_num, "aggregator", "aggregation_fail", f"Create instance error: {e}")
        return base_model

    # Zero out the aggregated state dict
    for key in aggregated_state_dict:
        aggregated_state_dict[key].zero_() # More explicit zeroing

    valid_updates_count = 0
    for update_info in participant_updates_list:
        participant_id = update_info.get('participantId', 'unknown_participant')
        model_update_str = update_info.get('modelUpdate')

        if not model_update_str:
            log_operation(log_dir, round_num, participant_id, "aggregation_skip", "Empty model update string")
            print(f" [Warning] Skipping empty model update from {participant_id}.")
            continue

        # Deserialize participant model state (not the full model object)
        # This uses the deserialize function which handles errors and returns None on failure
        # We don't need the full participant model object here, just its state_dict is implied
        # Let's refine this: deserialize should ideally just return the state_dict or None
        # Modify deserialize to potentially do this, or load into a temp model here.
        # Current approach: Load into temp model, extract state_dict.

        temp_model = deserialize_model(model_update_str, model_type, dataset_name)

        if temp_model is not None:
             participant_state_dict = temp_model.state_dict()
             # Add participant state dict to aggregated state dict
             for key in aggregated_state_dict:
                 if key in participant_state_dict:
                     aggregated_state_dict[key].add_(participant_state_dict[key].to(aggregated_state_dict[key].device))
                 else:
                     print(f" [Warning] Key '{key}' missing in update from {participant_id}.")
             valid_updates_count += 1
        else:
            # Deserialization failed (error already printed by deserialize_model)
            log_operation(log_dir, round_num, participant_id, "aggregation_skip", "Deserialization failed")
            print(f" [Warning] Skipping update from {participant_id} due to deserialization failure.")
            continue # Skip this update

    # Perform averaging
    if valid_updates_count > 0:
        print(f">> Averaging {valid_updates_count} valid updates...")
        for key in aggregated_state_dict:
            aggregated_state_dict[key].div_(valid_updates_count) # In-place division

        # Load the averaged state dict into the final aggregated model
        aggregated_model.load_state_dict(aggregated_state_dict)
        print(">> Aggregation finished.")
        log_operation(log_dir, round_num, "aggregator", "aggregation_success", f"Aggregated {valid_updates_count} models.")
        return aggregated_model
    else:
        print("[Warning] No valid participant updates processed. Returning base model.")
        log_operation(log_dir, round_num, "aggregator", "aggregation_skip", "No valid updates found.")
        return base_model