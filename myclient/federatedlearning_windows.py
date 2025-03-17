#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
from bcos3sdk.bcos3client import Bcos3Client
from client.contractnote import ContractNote
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client_config import client_config
import os
import traceback
from eth_utils import to_checksum_address
from eth_account.account import Account
import argparse
import shutil
import json
import torch
import datetime
from client_config import client_config as ClientConfig

# =====  导入 fl_multi_utils.py 中的通用函数和类  =====
from myclient.fl_windows_utils import (
    log_operation,
    load_mnist_data_partition_multi,
    load_mnist_test_data,
    load_cifar10_data_partition_multi,
    load_cifar10_test_data,
    MLP,
    CNN,
    serialize_model,
    deserialize_model,
    train_model,
    evaluate_model,
    upload_model_update,
    upload_global_model_hash,
    calculate_model_hash
)

# ----- 合约信息 (需要根据你的实际情况修改) -----
CONTRACT_NAME = "DecentralizedFederatedLearningHashWindow"
CONTRACT_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.sol"
ABI_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.abi"
BIN_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.bin"
CONTRACT_NOTE_NAME = "federatedlearning_window_demo"
CONTRACT_ADDRESS = "" #  !!! 合约地址硬编码在这里， 初始为空， 由第一个参与者部署 !!!

# ----- 配置 -----
demo_config = client_config

# ----- 自动编译合约 -----
if not (os.path.exists(ABI_PATH) and os.path.exists(BIN_PATH)):
    print(f"ABI or BIN file not found, compiling contract: {CONTRACT_PATH}")
    Compiler.compile_file(CONTRACT_PATH)

abi_file = ABI_PATH
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi


# ==========  参与者节点角色函数 (完全去中心化版本 -  包含合约部署, 本地评估) ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, participant_id="participant1", log_dir="fl_decentralized_log"): #  !!! 日志文件夹修改为 fl_decentralized_log !!!
    print(f"\n>>Starting Participant Node ({participant_id} - Decentralized, No Central Server)...") #  !!! 修改打印信息， 强调 No Central Server !!!
    print(f"\n>>>>> Federated Learning Round: {args.current_round} (Participant Node - Training & Evaluation) <<<<<<") #  !!! 修改打印信息，  包含 Evaluation !!!
    print(f"\n>>Participant Node ({participant_id}) in Round: {args.current_round} - Decentralized, Local Evaluation") #  !!! 修改打印信息，  强调 Local Evaluation !!!
    log_operation(log_dir, args.current_round, participant_id, "training_evaluation_node_start", f"Participant Node {participant_id} Training & Evaluation Node started in decentralized mode.") #  !!! 修改 operation_type !!!


    # ==========  合约部署 (由第一个参与者节点完成) ==========
    global CONTRACT_ADDRESS #  !!! 声明使用全局变量 CONTRACT_ADDRESS !!!
    if not CONTRACT_ADDRESS: #  !!!  如果 CONTRACT_ADDRESS 为空，  则进行部署 !!!
        print(f"\n>>Participant Node ({participant_id}): Deploying contract as the first participant...") #  !!! 修改打印信息，  强调由第一个参与者部署 !!!
        log_operation(log_dir, args.current_round, participant_id, "contract_deploy_start", f"Participant Node {participant_id} starts deploying contract.") #  !!! 修改 operation_type !!!
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        participant_ids_list = [f"participant{i+1}" for i in range(args.num_participants)]
        deploy_result = participant_client.deploy(contract_bin, constructor_args=[participant_ids_list, args.window_size]) #  !!!  部署合约时传递参与者 ID 列表和窗口大小 !!!
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            log_operation(log_dir, args.current_round, participant_id, "contract_deploy_fail", f"Participant Node {participant_id} contract deployment failed.") #  !!! 修改 operation_type !!!
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"] #  !!!  更新全局变量 CONTRACT_ADDRESS !!!
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
        print(f">>Participant Node ({participant_id}): Contract deployed successfully, contract address: {CONTRACT_ADDRESS}") #  !!! 修改打印信息 !!!
        log_operation(log_dir, args.current_round, participant_id, "contract_deploy_success", f"Participant Node {participant_id} contract deployed successfully, address: {CONTRACT_ADDRESS}") #  !!! 修改 operation_type !!!
    else: #  后续参与者节点直接使用已部署的合约地址
        print(f"\n>>Participant Node ({participant_id}): Using existing contract at address: {CONTRACT_ADDRESS}") #  !!! 修改打印信息 !!!
        log_operation(log_dir, args.current_round, participant_id, "contract_address_reuse", f"Participant Node {participant_id} reuses existing contract at address: {CONTRACT_ADDRESS}") #  !!! 修改 operation_type !!!


    # ----- 1.  获取所有参与者的模型更新 (用于本地全局模型计算) -----
    print(f"\n>>Participant Node ({participant_id}): Downloading All Participant Model Updates for Local Aggregation...")
    log_operation(log_dir, args.current_round, participant_id, "download_all_participant_updates_start", f"Participant Node {participant_id} starts downloading all participant model updates for local aggregation.")
    participant_updates_json_str = participant_client.call(contract_address, contract_abi, "getParticipantUpdatesForAggregation", [args.current_round])[0]
    if participant_updates_json_str:
        print(f"\n>>All Participant Model Updates downloaded successfully (for Participant: {participant_id}, JSON String).")
        log_operation(log_dir, args.current_round, participant_id, "download_all_participant_updates_success", f"Participant Node {participant_id} downloaded all participant model updates successfully for local aggregation.")
        participant_updates = json.loads(participant_updates_json_str) # JSON 解析
        print(f"\n>>[DEBUG - Participant Node {participant_id}] All Participant Updates (JSON): \n{json.dumps(participant_updates, indent=2)}") # DEBUG - 打印 JSON 数据
    else:
        print(f"\n>>Failed to download All Participant Model Updates (for Participant: {participant_id}).")
        log_operation(log_dir, args.current_round, participant_id, "download_all_participant_updates_fail", f"Participant Node {participant_id} failed to download all participant model updates for local aggregation.")
        print(f"\n>>Local Global Model Aggregation May be Inaccurate: No Participant Model Updates Downloaded.")
        log_operation(log_dir, args.current_round, participant_id, "local_aggregation_inaccurate_warning", f"Participant Node {participant_id} local global model aggregation may be inaccurate due to participant updates download failure.")
        participant_updates = [] #  即使下载失败， 也继续进行本地训练和哈希上传， 但聚合可能不准确


    # ----- 2. 加载本地模型 (初始模型) -----
    print(f"\n>>Participant Node ({participant_id}): Initializing Local Model (No Global Model Download in Decentralized Mode)...") #  !!! 修改打印信息，  强调不下载全局模型 !!!
    log_operation(log_dir, args.current_round, participant_id, "initialize_local_model_start", f"Participant Node {participant_id} initializes local model (no global model download in decentralized mode).")
    if args.model == 'cnn':
        model = CNN()
    else:
        model = MLP()
    log_operation(log_dir, args.current_round, participant_id, "initialize_local_model_success", f"Participant Node {participant_id} local model initialized.")


    # ----- 3. 本地模型训练 -----
    print(f"\n>>Participant Node ({participant_id}): Loading Training Data and Starting Local Model Training...")
    if args.dataset == 'cifar10':
        train_loader = load_cifar10_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants)
    else:
        train_loader = load_mnist_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants)
    log_operation(log_dir, args.current_round, participant_id, "local_training_start", f"Participant Node {participant_id} local model training started.")
    trained_model = train_model(model, train_loader, epochs=args.epochs, log_dir=log_dir, round_num=args.current_round, participant_id=participant_id) #  !!!  日志文件夹修改为 fl_decentralized_log !!!
    updated_model_str = serialize_model(trained_model)
    log_operation(log_dir, args.current_round, participant_id, "local_training_success", f"Participant Node {participant_id} local model training finished.")


    # ----- 4.  本地计算全局模型 (使用所有参与者的模型更新)  !!! -----
    print(f"\n>>Participant Node ({participant_id}): Locally Aggregating Global Model using Downloaded Updates...")
    log_operation(log_dir, args.current_round, participant_id, "local_aggregation_start", f"Participant Node {participant_id} starts local global model aggregation.")
    local_global_model = aggregate_global_model(trained_model, participant_updates, model_type=args.model) #  !!!  使用本地训练的模型和下载的更新进行聚合，得到本地全局模型 !!!
    local_global_model_str = serialize_model(local_global_model) #  !!!  序列化本地全局模型 !!!
    log_operation(log_dir, args.current_round, participant_id, "local_aggregation_success", f"Participant Node {participant_id} local global model aggregation finished.")


    # -----  !!!  5. 计算本地全局模型的哈希值  !!! -----
    print(f"\n>>Participant Node ({participant_id}): Calculating Hash of Locally Aggregated Global Model...")
    local_global_model_hash = calculate_model_hash(local_global_model) #  !!! 计算本地全局模型的哈希 !!!
    print(f"\n>>Participant Node ({participant_id}) - Local Global Model Hash: {local_global_model_hash.hex()}")
    log_operation(log_dir, args.current_round, participant_id, "calculate_local_global_model_hash_success", f"Participant Node {participant_id} local global model hash calculated successfully.")


    # ----- 6. 上传模型更新 (仍然保留上传模型更新的步骤， 用于信息共享和可能的未来扩展) -----
    print(f"\n>>Participant Node ({participant_id}): Uploading Model Update (for Information Sharing)...")
    log_operation(log_dir, args.current_round, participant_id, "upload_model_update_start", f"Participant Node {participant_id} starts uploading model update for information sharing.")
    if upload_model_update(participant_client, contract_abi, contract_address, updated_model_str, participant_id, args.current_round):
        print(f"\n>>Model update uploaded successfully (from Participant: {participant_id} - for Information Sharing).")
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_success", f"Participant Node {participant_id} model update uploaded successfully for information sharing.")
    else:
        print(f"\n>>Model update upload failed (from Participant: {participant_id} - for Information Sharing).")
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_fail", f"Participant Node {participant_id} model update upload failed for information sharing.")


    # -----  !!!  7. 上传本地全局模型哈希  !!! -----
    print(f"\n>>Participant Node ({participant_id}): Uploading Local Global Model Hash for Consensus...")
    log_operation(log_dir, args.current_round, participant_id, "upload_global_model_hash_start", f"Participant Node {participant_id} starts uploading local global model hash for consensus.")
    if upload_global_model_hash(participant_client, contract_abi, contract_address, local_global_model_hash, participant_id, args.current_round): #  !!! 上传本地全局模型哈希 !!!
        print(f"\n>>Local Global Model Hash uploaded successfully (from Participant: {participant_id} - for Consensus).")
        log_operation(log_dir, args.current_round, participant_id, "upload_global_model_hash_success", f"Participant Node {participant_id} local global model hash uploaded successfully for consensus.")
    else:
        print(f"\n>>Local Global Model Hash upload failed (from Participant: {participant_id} - for Consensus).")
        log_operation(log_dir, args.current_round, participant_id, "upload_global_model_hash_fail", f"Participant Node {participant_id} local global model hash upload failed for consensus.")


    # -----  !!!  8. 本地模型评估  !!! -----
    print(f"\n>>Participant Node ({participant_id}): Evaluating Local Global Model on Test Data...") #  !!! 修改打印信息，  强调本地评估 !!!
    log_operation(log_dir, args.current_round, participant_id, "local_evaluation_start", f"Participant Node {participant_id} starts local global model evaluation.") #  !!! 修改 operation_type !!!
    if args.dataset == 'cifar10':
        test_loader = load_cifar10_test_data()
    else:
        test_loader = load_mnist_test_data()
    evaluate_model(local_global_model, test_loader, log_dir=log_dir, round_num=args.current_round, role_name=participant_id) #  !!! 本地评估，  role_name 使用 participant_id,  日志文件夹修改为 fl_decentralized_log !!!
    log_operation(log_dir, args.current_round, participant_id, "local_evaluation_success", f"Participant Node {participant_id} local global model evaluation finished.") #  !!! 修改 operation_type !!!


    log_operation(log_dir, args.current_round, participant_id, "training_evaluation_node_complete", f"Participant Node {participant_id} Training & Evaluation Node Finished Round: {args.current_round} in decentralized mode.") #  !!! 修改 operation_type !!!
    print(f"\n>>Participant Node ({participant_id}) Finished Round: {args.current_round} - Decentralized, Local Evaluation") #  !!! 修改打印信息，  强调 Local Evaluation !!!



# ==========  主程序入口 (Demo 模式 -  多进程多客户端，多节点模拟) ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Pure Decentralized Hash Consensus Version, Multi-Node Version, No Central Server, Local Evaluation, Parameter Configurable, Multi-Round Training and Logging, Per-Round Local Evaluation, Clear Log Folder, Optimized Client Creation):------------------------------------") #  !!! 修改打印信息 -  标记为 Pure Decentralized Hash Consensus Version 和 No Central Server

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Pure Decentralized Hash Consensus, No Central Server, Multi-Client Simulation - Single Process, Multi-Process Structure, Parameter Configurable, Multi-Round Training and Logging, Per-Round Local Evaluation, Clear Log Folder, Optimized Client Creation - Pure Decentralized Hash Consensus Version, Multi-Node Version, Local Evaluation)") #  !!! 修改打印信息， 更清晰地表达当前模式 - Pure Decentralized Hash Consensus Version, No Central Server, Local Evaluation


    # ==========  参数解析 (简化 role 参数) ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script (Pure Decentralized Hash Consensus Version, Multi-Node Version, No Central Server, Local Evaluation)") #  !!! 修改 description - Pure Decentralized Hash Consensus Version, No Central Server, Local Evaluation !!!
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use (mlp or cnn)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round (default: 1)')
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds (default: 2)')
    parser.add_argument('--role', type=str, default='demo', choices=['demo', 'participant'], help='Role to run (demo, participant)') #  !!! 修改 role 参数 choices -  移除 coordinator !!!
    parser.add_argument('--log_dir', type=str, default='fl_decentralized_log', help='Directory to save log files (default: fl_decentralized_log)') #  !!!  日志文件夹参数修改为 fl_decentralized_log !!!
    parser.add_argument('--num_participants', type=int, default=3, help='Number of participant nodes to simulate in demo mode (default: 3)')
    parser.add_argument('--address', type=str, default='', help='Contract address to interact with')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for contract to store model updates (default: 5)')
    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Role: {args.role}, Log Dir: {args.log_dir}, Num Participants: {args.num_participants}, Window Size: {args.window_size}") #  !!!  打印配置信息 !!!


    role = args.role #  从命令行参数获取角色 (虽然 demo 模式固定为 demo)
    log_dir = args.log_dir #  从命令行参数获取日志文件夹路径
    num_participants = args.num_participants
    window_size = args.window_size


    # ========== 清空日志文件夹 (如果存在) ==========
    if os.path.exists(log_dir):
        print(f"\n>>Clearing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    else:
        print(f"\n>>Log directory not found, creating: {log_dir}")

    # ========== 创建日志文件夹 ==========
    os.makedirs(log_dir, exist_ok=True)


    participant_clients = [] #  !!!  只保留参与者客户端 !!!
    for i in range(num_participants):
        client_config=ClientConfig()
        client_config.account_keyfile = f"client{i}.keystore"
        client_config.account_password = f"{i}"*6
        participant_clients.append(Bcos3Client(client_config))


    # ==========  多轮联邦学习循环 (纯去中心化版本， 无需 Coordinator 节点) ==========
    print(f"\n>>Starting Pure Decentralized Federated Learning Rounds (No Central Server) ...") #  !!! 修改打印信息， 强调 Pure Decentralized 和 No Central Server !!!
    print(f"\n>>[DEBUG - Main Loop Start] Rounds: {args.rounds}") # DEBUG - 输出总轮数
    for round_num in range(1, args.rounds + 1): # 从第 1 轮循环到 rounds 轮
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}") # 添加轮次分隔符
        args.current_round = round_num # 将当前轮数添加到 args 中
        args.num_participants = num_participants
        print(f"\n>>[DEBUG - Main Loop - Round Start] Current Round: {args.current_round}, Num Participants: {args.num_participants}") # DEBUG - 输出每轮循环开始时的轮数


        # ========== 模拟参与者节点角色 (Demo Mode - Participant Part - Training, Aggregation, Upload Hash, Local Evaluation) -  所有步骤都在参与者节点完成 ==========
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training & Evaluation Nodes) process starting - Pure Decentralized Aggregation & Local Evaluation...") #  !!! 修改打印信息， 强调 Pure Decentralized Aggregation & Local Evaluation !!!
        print(f"\n>>[DEBUG - Participant Nodes Start] Round: {args.current_round}") # DEBUG - 输出 Participant Nodes 开始时的轮数

        # -----  模拟多个参与者节点并行训练 -----
        participant_ids = [f"participant{i+1}" for i in range(num_participants)]
        for i, participant_id in enumerate(participant_ids):
            print(f"\n>>Starting Participant Node ({participant_id})...")
            run_participant_node(participant_clients[i], contract_abi, CONTRACT_ADDRESS, args, participant_id=participant_id, log_dir=log_dir) #  !!!  所有步骤都在 run_participant_node 中完成 !!!

        print(f"\n>>[DEBUG - Participant Nodes End] Round: {args.current_round}") # DEBUG - 输出 Participant Nodes 结束时的轮数
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training & Evaluation Nodes) process finished - Pure Decentralized Aggregation & Local Evaluation.") #  !!! 修改打印信息， 强调 Pure Decentralized Aggregation & Local Evaluation !!!


    print("\n>>Federated Learning Demo (Pure Decentralized Hash Consensus Version, Multi-Node Version, No Central Server, Local Evaluation, Parameter Configurable, Multi-Round Training and Logging, Per-Round Local Evaluation, Clear Log Folder, Optimized Client Creation) Finished!") #  !!! 修改打印信息， 更清晰地表达当前模式