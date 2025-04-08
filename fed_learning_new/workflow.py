# -*- coding: utf-8 -*-
# Federated Learning round workflow logic. Python 3.7 compatible.

import json
import traceback
from typing import Dict, Any # For type hinting args

# Import functions from sibling modules
try:
    from . import communication as comm
    from . import serialization as serial
    from . import data_utils
    from . import training
    from . import aggregation
    from . import evaluation
    from .setup_utils import log_operation # Assuming log_operation is in setup_utils
    from .config import SERVER_ROLE_NAME
except ImportError as e:
    print(f"[ERROR][workflow] Failed to import sibling modules: {e}")
    # Define dummy functions or raise error if imports fail
    def log_operation(*args, **kwargs): pass
    class comm: download_global_model=None; upload_model_update=None; get_participant_updates_from_chain=None
    class serial: deserialize_model=None; serialize_model=None
    class data_utils: load_mnist_data_partition=None; load_cifar10_data_partition=None; load_mnist_test_data=None; load_cifar10_test_data=None
    class training: train_model=None
    class aggregation: aggregate_global_model=None
    class evaluation: evaluate_model=None
    SERVER_ROLE_NAME = "server"


# Using args object for parameters, define a placeholder type hint if needed
ArgsType = Any # Replace with a more specific type hint if using TypedDict or dataclass later


def run_participant_round(
    participant_client: Any, # Use Any if Bcos3Client import fails
    participant_id: str,
    participant_index: int,
    contract_abi: list,
    contract_address: str,
    args: ArgsType, # Pass the argparse namespace or similar object
    log_dir: str
    ) -> bool:
    """Runs the logic for one participant in one FL round."""
    round_num = args.current_round
    print(f"\n--- Running Participant: {participant_id} (Round: {round_num}) ---")
    log_operation(log_dir, round_num, participant_id, "round_start", f"P{participant_index+1} starting.")
    step_success = True

    # 1. Download Global Model
    global_model_str = comm.download_global_model(participant_client, contract_abi, contract_address)
    if global_model_str is None:
        log_operation(log_dir, round_num, participant_id, "download_fail", "Failed to get global model.")
        print(f"[ERROR] {participant_id} failed download.")
        return False

    # 2. Deserialize Model
    model_to_train = serial.deserialize_model(global_model_str, args.model, args.dataset)
    if model_to_train is None:
        log_operation(log_dir, round_num, participant_id, "deserialize_fail", "Failed.")
        print(f"[ERROR] {participant_id} failed deserialize.")
        return False

    # 3. Load Data Partition
    print(f">> Loading data for {participant_id}...")
    train_loader = None
    if args.dataset == 'mnist':
        train_loader = data_utils.load_mnist_data_partition(
            batch_size=args.batch_size,
            participant_index=participant_index,
            total_participants=args.num_participants
        )
    elif args.dataset == 'cifar10':
        train_loader = data_utils.load_cifar10_data_partition(
            batch_size=args.batch_size,
            participant_index=participant_index,
            total_participants=args.num_participants
        )
    # Add elif for other datasets here if needed

    if train_loader is None or len(train_loader.dataset) == 0:
         log_operation(log_dir, round_num, participant_id, "data_load_fail", "Loader None or empty.")
         print(f"[ERROR] Failed data load or partition empty for {participant_id}.")
         # Decide policy: fail or allow skip? Let's fail for now.
         return False

    # 4. Train Model
    # train_model now returns the model object regardless of internal errors
    trained_model = training.train_model(
        model=model_to_train,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_dir=log_dir,
        round_num=round_num,
        participant_id=participant_id,
        model_type=args.model
    )
    # We could add a check here if train_model indicated failure via logs or return value

    # 5. Serialize Trained Model
    trained_model_str = serial.serialize_model(trained_model)
    if not trained_model_str:
        log_operation(log_dir, round_num, participant_id, "serialize_fail", "Empty string.")
        print(f"[ERROR] {participant_id} failed serialization.")
        return False

    # 6. Upload Model Update
    step_success = comm.upload_model_update(
        client=participant_client,
        contract_abi=contract_abi,
        contract_address=contract_address,
        model_str=trained_model_str,
        role_name=participant_id,
        round_num=round_num,
        log_dir=log_dir # Pass log_dir
    )

    log_operation(log_dir, round_num, participant_id, "round_finish", f"Success: {step_success}")
    print(f"--- Participant {participant_id} finished round {round_num} (Success: {step_success}) ---")
    return step_success


def run_server_round(
    server_client: Any, # Use Any if Bcos3Client import fails
    contract_abi: list,
    contract_address: str,
    args: ArgsType,
    log_dir: str
    ) -> bool:
    """Runs the logic for the server in one FL round."""
    round_num = args.current_round
    print(f"\n--- Running Server (Round: {round_num}) ---")
    log_operation(log_dir, round_num, SERVER_ROLE_NAME, "round_start", "Server phase.")
    step_success = True

    # 1. Download Base Global Model
    base_model_str = comm.download_global_model(server_client, contract_abi, contract_address)
    if base_model_str is None:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "download_base_fail", "Failed.")
        print("[ERROR] Server failed download base model.")
        return False

    # 2. Deserialize Base Model
    base_model = serial.deserialize_model(base_model_str, args.model, args.dataset)
    if base_model is None:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "deserialize_base_fail", "Failed.")
        print("[ERROR] Server failed deserialize base model.")
        return False

    # 3. Download Participant Updates
    p_updates_json = comm.get_participant_updates_from_chain(
        server_client, contract_abi, contract_address, round_num
    )
    if p_updates_json is None:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "download_updates_fail", "Failed.")
        print(f"[Warning] Server failed download participant updates R{round_num}. Skipping agg.")
        # Fail the round if updates are crucial
        return False

    # 4. Parse Participant Updates
    p_updates_list = []
    try:
        parsed_obj = json.loads(p_updates_json)
        if isinstance(parsed_obj, list):
             p_updates_list = parsed_obj
             print(f"   Parsed {len(p_updates_list)} updates from JSON.")
        else:
             raise TypeError("Parsed JSON is not a list")
    except (json.JSONDecodeError, TypeError) as e:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "parse_updates_fail", f"Error: {e}")
        print(f"[ERROR] Server failed parse participant updates JSON: {e}")
        return False

    if not p_updates_list:
         log_operation(log_dir, round_num, SERVER_ROLE_NAME, "no_valid_updates", "Update list empty.")
         print(f"[Info] No participant updates for R{round_num}. Skipping aggregation & eval.")
         # Consider just uploading the base model again? Or return True to signify round passed trivially.
         # Let's return True, but the main loop might check logs later.
         return True


    # 5. Aggregate Models
    agg_model = aggregation.aggregate_global_model(
        base_model=base_model,
        participant_updates_list=p_updates_list,
        model_type=args.model,
        dataset_name=args.dataset,
        log_dir=log_dir,
        round_num=round_num
    )
    if agg_model is None:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "aggregation_fail", "Returned None.")
        print("[ERROR] Server model aggregation failed.")
        return False
    # Aggregation success is logged within aggregate_global_model

    # 6. Serialize Aggregated Model
    agg_model_str = serial.serialize_model(agg_model)
    if not agg_model_str:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "serialize_agg_fail", "Failed.")
        print("[ERROR] Server failed serialize aggregated model.")
        return False

    # 7. Upload Aggregated Model
    step_success = comm.upload_model_update(
        client=server_client,
        contract_abi=contract_abi,
        contract_address=contract_address,
        model_str=agg_model_str,
        role_name=SERVER_ROLE_NAME,
        round_num=round_num,
        log_dir=log_dir
    )
    if not step_success:
        print("[ERROR] Server failed upload aggregated model.")
        return False # Critical step failed
    log_operation(log_dir, round_num, SERVER_ROLE_NAME, "upload_global_success", "")

    # 8. Evaluate Aggregated Model
    print("\n>> Evaluating aggregated model...")
    test_loader = None
    if args.dataset == 'mnist':
        test_loader = data_utils.load_mnist_test_data(batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        test_loader = data_utils.load_cifar10_test_data(batch_size=args.batch_size)
    # Add elif for other datasets

    if test_loader is None:
        log_operation(log_dir, round_num, SERVER_ROLE_NAME, "test_data_fail", "Failed.")
        print("[ERROR] Server failed load test data.")
        return False

    eval_metric = evaluation.evaluate_model(
        model=agg_model,
        test_loader=test_loader,
        log_dir=log_dir,
        round_num=round_num,
        model_type=args.model
    )
    # Evaluation success/metric logged within evaluate_model

    log_operation(log_dir, round_num, SERVER_ROLE_NAME, "round_finish", f"Eval OK (Metric: {eval_metric:.4f})")
    print(f"--- Server finished round {round_num} ---")
    return True