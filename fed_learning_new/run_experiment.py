#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Main script to run FL experiments (Python 3.7 Compatible Structure).

import sys
import os
import argparse
import time
import traceback
import json

# --- [Ensure Project Root is in Path] ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # fed_learning_py37's parent
if project_root not in sys.path:
    print(f"[run_experiment.py] Adding project root to path: {project_root}")
    sys.path.insert(0, project_root)


# --- Import local modules from fed_learning_py37 ---
try:
    from . import config
    from . import setup_utils
    from . import workflow
except ImportError as e:
     print(f"[ERROR][run_experiment] Failed to import local modules: {e}")
     sys.exit(1)

# --- BCOS SDK Imports (for ABI loading/compilation) ---
try:
    from client.datatype_parser import DatatypeParser
    from client.common.compiler import Compiler
except ImportError as e:
    print(f"[ERROR][run_experiment] Failed to import BCOS SDK components: {e}")
    sys.exit(1)


# ========== Main Execution Logic ==========
def main():
    """Parses arguments and runs the federated learning simulation."""
    print("\n" + "="*60)
    print(" Starting Federated Learning Experiment (Python 3.7 Structure) ")
    print("="*60 + "\n")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Runner")
    # Use defaults from config module
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help="Dataset")
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help="Model architecture")
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS, help="Local epochs")
    parser.add_argument('--rounds', type=int, default=config.DEFAULT_ROUNDS, help="FL rounds")
    parser.add_argument('--num_participants', type=int, default=config.DEFAULT_PARTICIPANTS, help="# Participants")
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=config.DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument('--contract_address', type=str, default='', help="Existing contract address")
    parser.add_argument('--log_dir', type=str, default=config.DEFAULT_LOG_DIR, help="Log directory")

    args = parser.parse_args()

    print(">>> Experiment Configuration:")
    # Log config as JSON for easy parsing later
    config_log_msg = json.dumps(vars(args), indent=4)
    print(config_log_msg)
    print("-" * 30)

    # --- 1. Setup Logging ---
    if not setup_utils.setup_logging_directory(args.log_dir):
        print("[FATAL] Failed setup logging. Exiting.")
        sys.exit(1)
    setup_utils.log_operation(args.log_dir, 0, "System", "RunStart", f"Config: {vars(args)}")

    # --- 2. Load Contract ABI ---
    contract_abi = None
    try:
        # Attempt compilation if ABI not found (logic moved here from setup_utils)
        if not os.path.exists(config.ABI_PATH):
             print(f"ABI file not found: {config.ABI_PATH}. Compiling...")
             if not os.path.exists(config.CONTRACT_PATH):
                  print(f"[ERROR] Contract source not found: {config.CONTRACT_PATH}"); sys.exit(1)
             try:
                  Compiler.compile_file(config.CONTRACT_PATH, output_dir=config.CONTRACT_DIR)
                  print("   Compilation OK.")
                  if not os.path.exists(config.ABI_PATH):
                       print("[ERROR] ABI file still missing after compile!"); sys.exit(1)
             except Exception as compile_e:
                  print(f"[ERROR] Compilation failed: {compile_e}"); sys.exit(1)

        parser_abi = DatatypeParser()
        parser_abi.load_abi_file(config.ABI_PATH)
        contract_abi = parser_abi.contract_abi
        print("Contract ABI loaded successfully.")
        setup_utils.log_operation(args.log_dir, 0, "System", "LoadABI", "Success")
    except Exception as e:
        print(f"[FATAL] Failed load contract ABI: {e}")
        setup_utils.log_operation(args.log_dir, 0, "System", "LoadABI", f"Failed: {e}")
        traceback.print_exc(); sys.exit(1)

    # --- 3. Initialize Clients ---
    server_client = setup_utils.initialize_bcos_client(purpose="Server")
    if not server_client: sys.exit(1)

    # InitializeParticipantClients now returns Union[Tuple[List, List], Tuple[None,None]]
    result = setup_utils.initialize_participant_clients(args.num_participants)
    if result[0] is None: # Check if first element (clients list) is None
        setup_utils.cleanup_clients(server_client, [])
        sys.exit(1)
    participant_clients, participant_ids = result # Unpack if successful


    # --- 4. Setup Contract ---
    contract_address = setup_utils.setup_contract(server_client, args.contract_address, args.log_dir, contract_abi)
    if not contract_address:
        setup_utils.cleanup_clients(server_client, participant_clients)
        sys.exit(1)

    # --- 5. FL Rounds Loop ---
    print(f"\n{'='*20} Starting FL ({args.rounds} Rounds) {'='*20}")
    overall_start_time = time.time()
    successful_rounds = 0

    for round_i in range(1, args.rounds + 1):
        round_start_time = time.time()
        print(f"\n---------- Round {round_i}/{args.rounds} Starting ----------")
        args.current_round = round_i # Make current round available in args

        # Participant Phase (Sequential Execution)
        print(f"\n>>> Participant Phase <<<")
        participant_round_success = True
        for p_idx in range(args.num_participants):
            try:
                 if not workflow.run_participant_round(
                     participant_clients[p_idx],
                     participant_ids[p_idx],
                     p_idx,
                     contract_abi, contract_address, args, args.log_dir
                 ):
                     participant_round_success = False
                     print(f"[Warning] Participant {participant_ids[p_idx]} failed round {round_i}.")
                     # Optional: break here if one failure should stop the round
            except Exception as p_err:
                 print(f"[FATAL] Uncaught exception participant {participant_ids[p_idx]} R{round_i}: {p_err}")
                 setup_utils.log_operation(args.log_dir, round_i, participant_ids[p_idx], "Exception", str(p_err))
                 traceback.print_exc()
                 participant_round_success = False
                 # Optional: break on fatal error

        # Optional check: if participant_round_success is False, maybe skip server?
        if not participant_round_success:
             print(f"[Warning] Participant phase had failures in R{round_i}. Skipping Server.")
             setup_utils.log_operation(args.log_dir, round_i, "System", "SkipServer", "Participant failures")
             continue # Go to next round

        # Server Phase
        print(f"\n>>> Server Phase <<<")
        try:
            server_round_success = workflow.run_server_round(
                server_client, contract_abi, contract_address, args, args.log_dir
            )
            if server_round_success:
                successful_rounds += 1
            else:
                 print(f"[Warning] Server phase failed or skipped in round {round_i}.")
                 # Decide if this is critical
        except Exception as s_err:
            print(f"[FATAL] Uncaught exception server R{round_i}: {s_err}")
            setup_utils.log_operation(args.log_dir, round_i, config.SERVER_ROLE_NAME, "Exception", str(s_err))
            traceback.print_exc()
            # break # Stop all rounds on fatal server error?

        round_duration = time.time() - round_start_time
        print(f"---------- Round {round_i} Finished (Duration: {round_duration:.2f} sec) ----------")

    # --- 6. Final Summary ---
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    print(f"\n{'='*20} FL Experiment Finished {'='*20}")
    print(f">> Total Rounds Requested : {args.rounds}")
    print(f">> Successful Rounds      : {successful_rounds}")
    print(f">> Total Execution Time : {total_time:.2f} seconds")
    setup_utils.log_operation(args.log_dir, args.rounds, "System", "RunEnd", f"SuccessRounds:{successful_rounds}/{args.rounds}, TotalTime:{total_time:.2f}s")

    # --- 7. Cleanup ---
    setup_utils.cleanup_clients(server_client, participant_clients)

    print("\nExperiment complete.")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()