# -*- coding: utf-8 -*-
# Blockchain communication functions. Python 3.7 compatible.

import traceback
from typing import Union, List, Dict, Any # Use typing.Union

# --- BCOS SDK Imports ---
# Assume Bcos3Client type hint is needed, handle potential import error if run standalone
try:
    from bcos3sdk.bcos3client import Bcos3Client
except ImportError:
    Bcos3Client = Any # Fallback type hint


# Assumes setup_utils is available for logging
try:
    from .setup_utils import log_operation
except ImportError:
    # Provide a dummy logger if run standalone
    def log_operation(log_dir, round_num, role_name, op_type, msg):
        print(f"[Dummy Log] R:{round_num} {role_name} {op_type}: {msg}")


def upload_model_update(
    client: Bcos3Client,
    contract_abi: List[Dict[str, Any]], # More specific ABI type hint
    contract_address: str,
    model_str: str,
    role_name: str,
    round_num: int,
    log_dir: str # Pass log_dir for logging failures
) -> bool:
    """Uploads a model string to the smart contract."""
    fn_name = "updateModel"
    args = [model_str, round_num, role_name]
    try:
        receipt = client.sendRawTransaction(
            contract_address, contract_abi, fn_name, args
        )
        if receipt and receipt.get("status") == 0:
            tx_hash = receipt.get('transactionHash', 'N/A')
            print(f"  Upload Success ({role_name}, R{round_num}). TxHash: {tx_hash[:10]}...")
            return True
        else:
            status = receipt.get('status', 'N/A') if receipt else 'N/A'
            output = receipt.get('output', '') if receipt else ''
            error_msg = f"Upload Failed ({role_name}, R{round_num}). Status: {status}. Output: {output or 'N/A'}"
            print(f"  [ERROR] {error_msg}")
            log_operation(log_dir, round_num, role_name, "upload_fail", error_msg)
            return False
    except Exception as e:
        error_msg = f"Upload Exception ({role_name}, R{round_num}): {e}"
        print(f"  [ERROR] {error_msg}")
        log_operation(log_dir, round_num, role_name, "upload_exception", str(e))
        traceback.print_exc()
        return False

def download_global_model(
    client: Bcos3Client,
    contract_abi: List[Dict[str, Any]],
    contract_address: str
) -> Union[str, None]: # Use Union
    """Downloads the latest global model string from the smart contract."""
    fn_name = "getModel"
    try:
        result = client.call(contract_address, contract_abi, fn_name, [])
        if result and isinstance(result, (list, tuple)) and len(result) > 0:
            model_str = result[0]
            print(f"  Downloaded Global Model (len:{len(model_str)}).")
            # Optional: Check if it's the initial placeholder string from config
            # from .config import INITIAL_MODEL_STR (Avoid circular import if possible)
            # if model_str == INITIAL_MODEL_STR: ...
            return model_str
        else:
            print(f"  [ERROR] Failed download global model. Result: {result}")
            return None
    except Exception as e:
        print(f"  [ERROR] Download Global Model Exception: {e}")
        traceback.print_exc()
        return None

def get_participant_updates_from_chain(
    client: Bcos3Client,
    contract_abi: List[Dict[str, Any]],
    contract_address: str,
    round_num: int
) -> Union[str, None]: # Use Union
    """Downloads participant updates for a specific round as a JSON string."""
    fn_name = "getParticipantUpdates"
    try:
        round_num_int = int(round_num)
        args = [round_num_int]
        result = client.call(contract_address, contract_abi, fn_name, args)

        if result and isinstance(result, (list, tuple)) and len(result) > 0:
            json_str = result[0]
            print(f"  Downloaded Participant Updates JSON (Round {round_num}, len:{len(json_str)}).")
            if not (json_str.startswith('[') and json_str.endswith(']')):
                 print(f"  [Warning] Downloaded string doesn't look like JSON array: {json_str[:50]}...")
            return json_str
        else:
            print(f"  [ERROR] Failed download participant updates (Round {round_num}). Result: {result}")
            return None
    except ValueError:
         print(f"  [ERROR] Invalid round number format: {round_num}")
         return None
    except Exception as e:
        print(f"  [ERROR] Download Participant Updates Exception (Round {round_num}): {e}")
        traceback.print_exc()
        return None