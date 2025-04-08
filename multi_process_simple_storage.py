#!/usr/bin/env python
# -*- coding: utf-8 -*-
# multi_process_simple_storage.py - Multiple processes calling SimpleStorage.set()

import sys
import os
import time
import multiprocessing
import argparse
import traceback
import random

# --- [Ensure Project Root is in Path] ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir # Assuming script is in project root
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# --- BCOS SDK Imports ---
try:
    from bcos3sdk.bcos3client import Bcos3Client
    from client.contractnote import ContractNote
    from client_config import client_config as ClientConfig
    from client.datatype_parser import DatatypeParser
    from client.common.compiler import Compiler
except ImportError as e:
    print(f"[ERROR] Failed to import BCOS SDK components: {e}")
    print("        Ensure bcos3sdk and client directories are accessible, and client_config.py exists.")
    sys.exit(1)

# --- Contract Configuration ---
CONTRACT_DIR = os.path.join(project_root, "contracts")
CONTRACT_NAME = "SimpleStorage" # <--- Changed Contract Name
ABI_PATH = os.path.join(CONTRACT_DIR, f"{CONTRACT_NAME}.abi")
BIN_PATH = os.path.join(CONTRACT_DIR, f"{CONTRACT_NAME}.bin")
CONTRACT_NOTE_NAME = "simple_storage_demo" # <--- Changed Note Name

# ========== Worker Function for Each Process ==========

def storage_worker(
    participant_index: int,
    contract_address: str,
    contract_abi: list,
    base_key_dir: str,
    ):
    """
    Function executed by each process to call SimpleStorage.set().
    """
    process_id = os.getpid()
    worker_id = f"Worker-{participant_index + 1}" # Use a more generic term
    print(f"[PID:{process_id}] Starting {worker_id}...")

    # --- 1. Configure and Initialize BCOS Client ---
    client = None # Initialize client to None for finally block
    try:
        client_config_obj = ClientConfig()
        key_file_path = f'client{participant_index}.keystore' # Example path, adjust as needed
        # Generate password based on index (ensure this matches your key generation)
        key_pass = f"{participant_index}" * 6


        setattr(client_config_obj, 'account_keyfile', key_file_path)
        setattr(client_config_obj, 'account_password', key_pass)

        client = Bcos3Client(client_config_obj)
        #client_address = client.account.address
        #print(f"[PID:{process_id}] {worker_id} client initialized. Address: {client_address}")

    except Exception as e:
        print(f"[PID:{process_id}][ERROR] Failed to initialize client for {worker_id}: {e}")
        traceback.print_exc()
        return

    # --- 2. Prepare Data for the 'set' function ---
    # Each process will try to set a unique value (e.g., based on index or PID)
    value_to_set = random.randint(100, 999) + participant_index * 1000 # Example: generate somewhat unique value
    print(f"[PID:{process_id}] {worker_id} preparing to set value: {value_to_set}")

    # --- 3. Interact with the Smart Contract ---
    fn_name = "set" # Function to call
    args = [value_to_set] # Argument for the 'set' function

    print(f"[PID:{process_id}] {worker_id} calling {fn_name}({value_to_set})...")
    try:
        receipt = client.sendRawTransaction(contract_address, contract_abi, fn_name, args)

        if receipt and receipt.get("status") == 0:
            tx_hash = receipt.get("transactionHash", "N/A")
            block_num = receipt.get("blockNumber", "N/A")
            print(f"[PID:{process_id}] {worker_id} transaction SUCCESS.")
            print(f"    Value Set: {value_to_set}, TxHash: {tx_hash}, Block: {block_num}")
        else:
            status = receipt.get("status", "N/A") if receipt else "N/A"
            output = receipt.get("output", "") if receipt else "N/A"
            print(f"[PID:{process_id}] {worker_id} transaction FAILED.")
            print(f"    Status: {status}, Output: {output or 'N/A'}")

    except Exception as e:
        print(f"[PID:{process_id}][ERROR] Exception during contract call for {worker_id}: {e}")
        traceback.print_exc()

    # --- 4. Cleanup Client ---
    finally:
        if client: # Check if client was successfully initialized
            try:
                client.finish()
                print(f"[PID:{process_id}] {worker_id} client finished.")
            except Exception as e:
                print(f"[PID:{process_id}][Warning] Error finishing client for {worker_id}: {e}")

# ========== Main Process Logic ==========

def main():
    """Sets up and runs the multi-process SimpleStorage interaction."""
    print("\n" + "="*50)
    print(" Multi-Process SimpleStorage Interaction Demo ")
    print("="*50 + "\n")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Multi-process SimpleStorage demo.")
    parser.add_argument('--num_workers', type=int, default=3, help="Number of worker processes to spawn")
    parser.add_argument('--contract_address', type=str, default='', help="Address of deployed SimpleStorage contract (deploy new if empty)")
    parser.add_argument('--key_dir', type=str, default=project_root, help="Directory containing client keystore files")
    args = parser.parse_args()

    print(">>> Configuration:")
    print(f"    Number of Workers: {args.num_workers}")
    print(f"    Contract Address : {args.contract_address or 'Deploy New'}")
    print(f"    Keystore Directory: {args.key_dir}")
    print("-" * 30)

    # --- 1. Load Contract ABI ---
    contract_abi = None
    try:
        if not os.path.exists(ABI_PATH):
             print(f"ABI file not found: {ABI_PATH}. Attempting compilation...")
             contract_sol_path = os.path.join(CONTRACT_DIR, f"{CONTRACT_NAME}.sol")
             if not os.path.exists(contract_sol_path):
                  print(f"[ERROR] Contract source file {contract_sol_path} not found for compilation.")
                  sys.exit(1)
             try:
                  Compiler.compile_file(contract_sol_path, output_path=CONTRACT_DIR)
                  print("   Compilation successful.")
                  if not os.path.exists(ABI_PATH): raise FileNotFoundError("ABI still not found")
             except Exception as compile_e:
                  print(f"[ERROR] Compilation failed: {compile_e}")
                  sys.exit(1)

        parser_abi = DatatypeParser()
        parser_abi.load_abi_file(ABI_PATH)
        contract_abi = parser_abi.contract_abi
        print(f"Contract ABI for {CONTRACT_NAME} loaded successfully.")
    except Exception as e:
        print(f"[FATAL] Failed to load contract ABI from {ABI_PATH}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Deploy Contract (if needed) ---
    target_contract_address = args.contract_address
    if not target_contract_address:
        print("\nDeploying new SimpleStorage contract...")
        deployer_client = None
        try:
            # Using default client config for deployment
            deployer_client = Bcos3Client()
            #print(f"Deployer client initialized (Address: {deployer_client.account.address}).")

            if not os.path.exists(BIN_PATH):
                 print(f"[ERROR] Contract binary file {BIN_PATH} not found. Cannot deploy.")
                 sys.exit(1)
            with open(BIN_PATH, 'r') as f: contract_bin = f.read()

            deploy_receipt = deployer_client.deploy(contract_bin)
            if deploy_receipt and deploy_receipt.get("status") == 0:
                target_contract_address = deploy_receipt["contractAddress"]
                ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, target_contract_address)
                print(f"SimpleStorage contract deployed successfully: {target_contract_address}")
            else:
                status = deploy_receipt.get("status", "N/A") if deploy_receipt else "N/A"
                output = deploy_receipt.get("output", "") if deploy_receipt else "N/A"
                print(f"[FATAL] SimpleStorage deployment failed. Status: {status}, Output: {output or 'N/A'}")
                sys.exit(1)
        except Exception as e:
            print(f"[FATAL] Exception during SimpleStorage deployment: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            if deployer_client:
                deployer_client.finish()
                print("Deployer client finished.")
    else:
        print(f"\nUsing existing SimpleStorage contract: {target_contract_address}")

    # --- 3. Spawn Worker Processes ---
    print(f"\nSpawning {args.num_workers} worker processes...")
    processes = []
    start_time = time.time()

    for i in range(args.num_workers):
        process = multiprocessing.Process(
            target=storage_worker,
            args=(
                i, # participant_index
                target_contract_address,
                contract_abi,
                args.key_dir,
            )
        )
        processes.append(process)
        process.start()
        print(f"  Process started for worker {i+1}.")
        # Optional: Add a small delay between starting processes if needed
        # time.sleep(0.1)

    # --- 4. Wait for all processes to complete ---
    print("\nWaiting for worker processes to finish...")
    successful_workers = 0
    for i, process in enumerate(processes):
        process.join() # Wait for process to terminate
        # Check exit code (0 typically means success)
        if process.exitcode == 0:
             successful_workers += 1
             # print(f"  Process for worker {i+1} finished successfully.")
        else:
             print(f"  [Warning] Process for worker {i+1} finished with exit code {process.exitcode}.")

    end_time = time.time()
    print(f"\nAll {args.num_workers} processes completed in {end_time - start_time:.2f} seconds.")
    print(f"({successful_workers} workers likely finished without fatal errors).")
    print("\nDemo finished.")

    # --- 5. Optional: Query final state ---
    print("\nQuerying final state of the contract...")
    query_client = None
    try:
        query_client = Bcos3Client() # Use default client for query
        final_value = query_client.call(target_contract_address, contract_abi, "get", [])
        last_sender = query_client.call(target_contract_address, contract_abi, "getLastSender", [])
        if final_value is not None and last_sender is not None:
             print(f"  Final value stored: {final_value[0]}")
             print(f"  Last sender was : {last_sender[0]}")
        else:
             print("  Failed to query final state.")
    except Exception as e:
        print(f"  Error querying final state: {e}")
    finally:
        if query_client:
             query_client.finish()


# --- Script Entry Point ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()