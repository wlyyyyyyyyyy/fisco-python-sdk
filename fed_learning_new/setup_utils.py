# -*- coding: utf-8 -*-
# Setup utilities: logging, BCOS clients, contract. Python 3.7 compatible.

import os
import shutil
import datetime
import traceback
from typing import Union, List, Tuple, Dict, Any

# --- BCOS SDK Imports ---
try:
    import sys
    project_root_setup = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root_setup not in sys.path:
        sys.path.insert(0, project_root_setup)

    from bcos3sdk.bcos3client import Bcos3Client
    from client.contractnote import ContractNote

    # --- [关键] 正确导入 ClientConfig ---
    # 检查 client_config.py 是否存在以及如何导入配置
    # 常见的模式是 client_config.py 直接定义一个名为 client_config 的对象
    if os.path.exists(os.path.join(project_root_setup, "client_config.py")):
        try:
            from client_config import client_config as ClientConfig_obj
            print("[INFO][setup_utils] Imported 'client_config' object as ClientConfig_obj.")
            # 如果 ClientConfig_obj 确实是一个配置对象，我们可以直接用它
            # 如果它是一个类，需要实例化 ClientConfig_obj()
            # 这里假设它是对象，如果报错是类，需要调整为 ClientConfig_obj()
            DefaultConfigForClient = ClientConfig_obj
        except ImportError:
             print("[ERROR][setup_utils] Found client_config.py but failed to import 'client_config' object.")
             DefaultConfigForClient = None
    else:
        print("[Warning][setup_utils] client_config.py not found. BCOS client might use internal defaults if possible.")
        DefaultConfigForClient = None # 表明无法加载默认配置


    from client.datatype_parser import DatatypeParser
    from client.common.compiler import Compiler
except ImportError as e:
    print(f"[ERROR][setup_utils] Failed to import BCOS SDK components: {e}")
    Bcos3Client = Any
    DefaultConfigForClient = None # 导入失败，无法使用默认配置
    DatatypeParser = Any
    Compiler = Any


# Import config constants
try:
    from .config import (
        DEFAULT_KEYSTORE_DIR, CONTRACT_NAME, CONTRACT_NOTE_NAME,
        BIN_PATH, CONTRACT_PATH, CONTRACT_DIR
    )
except ImportError:
    # Fallback constants
    # ... (之前的 fallback 定义) ...
    DEFAULT_KEYSTORE_DIR = "."
    CONTRACT_NAME = "EnhancedFederatedLearning"
    # ... (其他 fallback) ...

# ========== Logging Utility ==========
# ... (log_operation 函数不变) ...
def log_operation(log_dir: str, round_num: int, role_name: str, operation_type: str, message: str):
    """Logs an operation to a central log file."""
    log_filename = os.path.join(log_dir, f"fl_operations_log.txt")
    try:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_info = (
            f'{timestamp} | R:{round_num:<3d} | Role:{role_name:<12} | '
            f'Op:{operation_type:<20} | Msg:{message}'
        )
        with open(log_filename, 'a') as log_file:
            log_file.write(log_info + '\n')
    except Exception as e:
        print(f"[ERROR] Failed to write log to {log_filename}: {e}")


# ========== Directory Setup Utility ==========
# ... (setup_logging_directory 函数不变) ...
def setup_logging_directory(log_dir: str) -> bool:
    """Cleans up (if exists) and creates the logging directory."""
    try:
        if os.path.exists(log_dir):
            print(f"\n>> Clearing existing log directory: {log_dir}")
            shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f">> Log directory created/cleaned: {log_dir}")
        log_operation(log_dir, 0, "System", "LogInit", f"Log directory set to {log_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create log directory {log_dir}: {e}")
        return False


# ========== BCOS Client Setup ==========

def initialize_bcos_client(config_obj: Any = None, purpose: str = "Generic") -> Union[Bcos3Client, None]:
    """Initializes a Bcos3Client instance, ensuring a config object is used."""
    print(f">> Initializing BCOS Client ({purpose})...")
    try:
        # --- [修改开始] ---
        effective_config = config_obj
        if effective_config is None:
            print("   No specific config provided, using default config from client_config.py...")
            if DefaultConfigForClient is None:
                 print("[ERROR] Default configuration (ClientConfig_obj) failed to load or is not available.")
                 # 尝试不带参数初始化，让 Bcos3Client 自己尝试最后一次加载
                 # 这通常在 SDK 内部查找 client_config.py
                 print("   Attempting Bcos3Client(None) as last resort...")
                 effective_config = None
                 # 注意：如果这里 Bcos3Client(None) 仍然失败并报同样的错，
                 # 说明 SDK 无法找到或解析 client_config.py
            else:
                # 假设 DefaultConfigForClient 是我们成功导入的对象
                 effective_config = DefaultConfigForClient
                 print("   Using the imported default config object.")

        # --- [修改结束] ---

        # 现在调用 Bcos3Client，effective_config 要么是传入的，要么是默认对象，要么是 None (最后尝试)
        # 注意：如果传入的是 None，且 SDK 内部加载也失败，下面这行仍然会报错
        client = Bcos3Client(effective_config)

        # --- 连接检查 ---
        try:
             block_num = client.getBlockNumber()
             print(f"   Connection check OK (Current block: {block_num}).")
        except Exception as conn_e:
             # 根据SDK具体行为，初始化成功但无法连接是可能的
             print(f"[Warning] BCOS Client ({purpose}) initialized but connection check failed: {conn_e}")
             print(f"          Check network settings in config and node status.")
             # 不在此处返回 None，让调用者决定是否继续

        print(f">> BCOS Client ({purpose}) initialized successfully.")
        return client

    # --- 错误处理 ---
    except AttributeError as ae:
         # 捕获特定的 'NoneType' object has no attribute 'crypto_type' 错误
         if 'crypto_type' in str(ae):
              print(f"[ERROR] BCOS Client ({purpose}) init failed: Configuration object is likely None or incomplete (missing 'crypto_type').")
              print(f"        Ensure 'client_config.py' exists, is correctly formatted, and accessible in PYTHONPATH.")
         else:
              # 其他 AttributeError
              print(f"[ERROR] Failed to initialize BCOS Client ({purpose}) - AttributeError: {ae}")
         traceback.print_exc()
         return None
    except Exception as e:
        # 其他初始化期间的异常
        print(f"[ERROR] Failed to initialize BCOS Client ({purpose}): {e}")
        if hasattr(config_obj, 'account_keyfile'): # 检查传入的 config_obj
             print(f"        Using Keyfile: {getattr(config_obj, 'account_keyfile', 'N/A')}")
        elif effective_config is not None: # 检查是否使用了默认配置
             print(f"        Used default config object.")
        else: # 如果是 None 传入且默认加载失败
             print(f"        Attempted initialization with None config after default load failure.")
        traceback.print_exc()
        return None

def initialize_participant_clients(num_participants: int) -> Union[Tuple[List[Bcos3Client], List[str]], Tuple[None, None]]:
    """Initializes clients for all participants using the correct approach."""
    print(f"\n>> Initializing {num_participants} Participant BCOS clients...")
    clients = []
    ids = []
    keystore_base_dir = DEFAULT_KEYSTORE_DIR

    for i in range(num_participants):
        participant_id = f"participant{i+1}"
        ids.append(participant_id)
        p_client = None
        try:
            # --- [与你提供的代码一致的逻辑] ---
            # 1. 获取/创建基础配置对象
            #    需要确定 DefaultConfigForClient 是对象还是类
            #    假设它是对象，我们需要能修改它的属性而不影响其他客户端
            #    如果它是类，我们需要实例化：config_base = DefaultConfigForClient()
            #    如果它是对象，安全的做法是深度拷贝？或者 BcosClient 内部会拷贝？
            #    最常见模式是 client_config.py 定义一个 client_config 对象，直接用
            if DefaultConfigForClient is None:
                print(f"[ERROR] Cannot initialize {participant_id}: Default config not loaded.")
                cleanup_clients(None, clients)
                return None, None
            # 假设 DefaultConfigForClient 是可以直接使用的配置对象
            # 如果需要实例化，应改为: client_config_obj = DefaultConfigForClient()
            # *** 为了安全，我们假设需要一个独立的配置对象，尝试实例化 ***
            # *** 如果 client_config.py 定义的是对象而非类，这里会报错，需要改回直接赋值 ***
            try:
                # 假设 client_config 模块里有一个名为 ClientConfig 的类
                # 如果你的文件是 client_config.py 定义了 client_config 对象，需要改成:
                # import copy; client_config_obj = copy.deepcopy(DefaultConfigForClient) # 深度拷贝以防修改影响全局
                # 或者，如果 Bcos3Client 内部会处理配置拷贝，可以直接用：
                client_config_obj = DefaultConfigForClient # 直接使用导入的对象
                # 如果 client_config.py 定义了类 ClientConfig, 则用下面的：
                # from client_config import ClientConfig as ClientConfigClass
                # client_config_obj = ClientConfigClass()
            except NameError: # Handle case where ClientConfigClass is not defined
                print("[Warning] Failed to determine how to get/instantiate ClientConfig. Using the imported object directly.")
                client_config_obj = DefaultConfigForClient # Fallback to using the object


            # 2. 设置特定账户信息
            key_file_name = f"client{i}.keystore"
            key_file_path = os.path.join(keystore_base_dir, key_file_name)
            key_pass = f"{i}"*6

            if not os.path.exists(key_file_path):
                print(f"[ERROR] Keystore file not found for {participant_id}: {key_file_path}")
                cleanup_clients(None, clients)
                return None, None

            # --- 修改配置对象的账户信息 ---
            # 使用 setattr 避免假设属性一定存在
            setattr(client_config_obj, 'account_keyfile', key_file_path) # 推荐使用完整路径
            setattr(client_config_obj, 'account_password', key_pass)
            # 如果 SDK 需要分别设置路径和文件名
            # setattr(client_config_obj, 'account_keyfile_path', keystore_base_dir)
            # setattr(client_config_obj, 'account_keyfile', key_file_name)

            # 3. 初始化客户端，传入配置好的对象
            p_client = initialize_bcos_client(client_config_obj, purpose=participant_id)
            # --- [逻辑结束] ---

            if p_client is None:
                # 初始化失败，错误已由 initialize_bcos_client 打印
                cleanup_clients(None, clients)
                return None, None
            clients.append(p_client)

        except Exception as e:
            print(f"[ERROR] Unexpected error initializing client for {participant_id}: {e}")
            traceback.print_exc()
            cleanup_clients(None, clients)
            return None, None

    print(">> All participant clients initialized.")
    return clients, ids


# ========== Contract Setup ==========
# ... (setup_contract 函数不变) ...
def setup_contract(
    server_client: Bcos3Client,
    contract_address_arg: str,
    log_dir: str,
    contract_abi: List[Dict[str, Any]]
) -> Union[str, None]:
    """Deploys a new contract or connects to an existing one."""
    contract_address = contract_address_arg
    if not contract_address:
        print(f"\n>> Deploying new contract {CONTRACT_NAME}...")
        try:
            if not os.path.exists(BIN_PATH):
                 print(f"   Contract binary file not found: {BIN_PATH}. Compiling...")
                 if not os.path.exists(CONTRACT_PATH):
                      print(f"[ERROR] Contract source file not found: {CONTRACT_PATH}")
                      return None
                 try:
                      Compiler.compile_file(CONTRACT_PATH, output_dir=CONTRACT_DIR)
                      print("   Compilation successful.")
                      if not os.path.exists(BIN_PATH):
                           print("[ERROR] BIN file still not found after compilation!")
                           return None
                 except Exception as compile_e:
                      print(f"[ERROR] Contract compilation failed: {compile_e}")
                      return None

            with open(BIN_PATH, 'r') as f: contract_bin = f.read()
            deploy_receipt = server_client.deploy(contract_bin)

            if deploy_receipt and deploy_receipt.get("status") == 0:
                contract_address = deploy_receipt["contractAddress"]
                ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, contract_address)
                print(f">> Contract deployed successfully: {contract_address}")
                log_operation(log_dir, 0, "deployer", "deploy_success", f"Addr:{contract_address}")
                return contract_address
            else:
                status = deploy_receipt.get('status', 'N/A') if deploy_receipt else 'N/A'
                output = deploy_receipt.get('output', '') if deploy_receipt else ''
                print(f"[ERROR] Contract deployment failed. Status:{status}, Output:{output}")
                log_operation(log_dir, 0, "deployer", "deploy_fail", f"Status:{status}, Output:{output}")
                return None
        except FileNotFoundError:
             print(f"[ERROR] Contract binary file not found: {BIN_PATH}")
             return None
        except Exception as e:
            print(f"[ERROR] Contract deployment exception: {e}")
            log_operation(log_dir, 0, "deployer", "deploy_exception", str(e))
            traceback.print_exc()
            return None
    else:
        print(f"\n>> Using existing contract: {contract_address}")
        print("   Verifying connection...")
        try:
            result = server_client.call(contract_address, contract_abi, "getModel", [])
            if result is not None:
                print(f"   Connection verified successfully.")
                log_operation(log_dir, 0, "system", "use_existing_contract", f"Addr:{contract_address}")
                return contract_address
            else:
                print(f"[ERROR] Failed to call getModel on existing contract {contract_address}.")
                log_operation(log_dir, 0, "system", "verify_contract_fail", f"getModel returned None for {contract_address}")
                return None
        except Exception as e:
            print(f"[ERROR] Exception while verifying existing contract: {e}")
            log_operation(log_dir, 0, "system", "verify_contract_exception", str(e))
            traceback.print_exc()
            return None


# ========== Client Cleanup ==========
# ... (cleanup_clients 函数不变) ...
def cleanup_clients(server_client: Union[Bcos3Client, None], participant_clients: List[Bcos3Client]):
    """Closes connections for all BCOS clients."""
    print("\n>> Closing BCOS client connections...")
    closed_count = 0
    # ... (rest of the function body) ...
    if server_client:
        try:
            server_client.finish()
            print("   Server client closed.")
            closed_count += 1
        except Exception as e:
            print(f"[Warning] Error closing server client: {e}")

    if participant_clients is None:
         participant_clients = []

    for i, p_client in enumerate(participant_clients):
        if p_client:
            try:
                p_client.finish()
                closed_count += 1
            except Exception as e:
                print(f"[Warning] Error closing participant {i+1} client: {e}")
    print(f">> Closed {closed_count} client connections.")