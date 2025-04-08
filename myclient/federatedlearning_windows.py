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

# =====  导入 fl_windows_utils.py 中的通用函数和类  =====
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
    submit_local_global_model_hash,
    aggregate_global_model_decentralized,
)

# ----- 合约信息 (需要根据你的实际情况修改) -----
CONTRACT_NAME = "DecentralizedFederatedLearningHashWindow"
CONTRACT_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.sol"
ABI_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.abi"
BIN_PATH = "./contracts/DecentralizedFederatedLearningHashWindow.bin"
CONTRACT_NOTE_NAME = "federatedlearning_multi_demo"
CONTRACT_ADDRESS = ""

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


# ==========  中央服务器角色函数 (多节点版本) ==========
def run_central_server(central_server_client, contract_abi, contract_address, args, run_mode, participant_clients=None, log_dir="fl_multi_log"):
    if run_mode == 0:
        print(f"\n>>Starting Central Server (Deployment and Setup)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Deployment) <<<<<<")
        print(f"\n>>Central Server (Deployment and Setup) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "deployment_start", "Central Server Deployment and Setup started.")

        global CONTRACT_ADDRESS
        if not CONTRACT_ADDRESS:
            print(f"\n>>Demo Mode - Deploying contract using the first client (index 0)...")
            deploying_client = participant_clients[0]
            with open(BIN_PATH, 'r') as f:
                contract_bin = f.read()
                f.close()
            deploy_result = deploying_client.deploy(contract_bin)
            if deploy_result is None or deploy_result["status"] != 0:
                print(f"Deploy contract failed, result: {deploy_result}")
                exit()
            CONTRACT_ADDRESS = deploy_result["contractAddress"]
            print(f"Demo Mode - Deploy contract success, contract address: {CONTRACT_ADDRESS}")
            ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)

            # ========== 设置初始参数 (numParticipants, windowSize) ==========
            print(f"\n>>Demo Mode - Setting initial contract parameters...")
            try:
                tx_hash_participants = central_server_client.sendRawTransaction(
                    CONTRACT_ADDRESS,
                    contract_abi,
                    "setNumParticipants",
                    [args.num_participants - 1 if args.num_participants > 0 else 0]
                )
                print(f"Demo Mode - Sent transaction to set numParticipants. Transaction hash: {tx_hash_participants}")
                receipt_participants = central_server_client.getTransactionReceipt(tx_hash_participants)
                print(f"Demo Mode - Receipt for setting numParticipants: {receipt_participants}")
                if not (receipt_participants and receipt_participants['status'] == 0):
                    print(f"Demo Mode - Failed to set numParticipants.")
                    exit()

                tx_hash_window = central_server_client.sendRawTransaction(
                    CONTRACT_ADDRESS,
                    contract_abi,
                    "setWindowSize",
                    [args.window_size if hasattr(args, 'window_size') else 5]
                )
                print(f"Demo Mode - Sent transaction to set windowSize. Transaction hash: {tx_hash_window}")
                receipt_window = central_server_client.getTransactionReceipt(tx_hash_window)
                print(f"Demo Mode - Receipt for setting windowSize: {receipt_window}")
                if not (receipt_window and receipt_window['status'] == 0):
                    print(f"Demo Mode - Failed to set windowSize.")
                    exit()

            except Exception as e:
                print(f"Demo Mode - Failed to send transaction to set initial contract parameters: {e}")
                exit()
        else:
            print(f"\n>>Demo Mode - Using existing contract at address: {CONTRACT_ADDRESS}")

        log_operation(log_dir, args.current_round, "server", "deployment_complete", f"Central Server Deployment and Setup Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Deployment and Setup) Completed Round: {args.current_round}")

    elif run_mode == 1:
        print(f"\n>>Starting Central Server (Preparation Node)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Preparation) <<<<<<")
        print(f"\n>>Central Server (Preparation Node) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "preparation_start", "Central Server Preparation Node started.")
        log_operation(log_dir, args.current_round, "server", "preparation_complete", f"Central Server Preparation Node Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Preparation Node) Completed Round: {args.current_round}")

    elif run_mode == 2:
        print(f"\n>>Starting Central Server (Participant Registration)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Registration) <<<<<<")
        print(f"\n>>Central Server (Participant Registration) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "registration_start", "Central Server Participant Registration started.")

        for i in range(1, args.num_participants):
            participant_address = participant_clients[i].account.address
            try:
                tx_hash = central_server_client.sendRawTransaction(
                    contract_address,
                    contract_abi,
                    "registerParticipant",
                    [participant_address]
                )
                print(f"Demo Mode - Sent transaction to register Participant {i}. Transaction hash: {tx_hash}")
                receipt = central_server_client.getTransactionReceipt(tx_hash)
                print(f"Demo Mode - Receipt for registering Participant {i}: {receipt}")
                if receipt and receipt['status'] == 0:
                    print(f"Demo Mode - Participant {i} registered successfully.")
                else:
                    print(f"Demo Mode - Failed to register participant {i}.")
                    exit()
            except Exception as e:
                print(f"Demo Mode - Failed to register participant {i}: {e}")
                exit()

        log_operation(log_dir, args.current_round, "server", "registration_complete", f"Central Server Participant Registration Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Participant Registration) Completed Round: {args.current_round}")

    elif run_mode == 3:
        print(f"\n>>Starting Central Server (Hash Checking and Next Round)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Hash Check) <<<<<<")
        print(f"\n>>Central Server (Hash Checking and Next Round) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "hash_check_start", f"Central Server Hash Checking started for Round {args.current_round}.")

        try:
            hashes = central_server_client.call(contract_address, contract_abi, "getAllLocalGlobalModelHashes", [args.current_round])[0]
            print(f"\n>>[DEBUG - Aggregation Node] Retrieved Hashes: {hashes}")
            if hashes:
                first_hash = hashes[0]
                all_same = all(hash_val == first_hash for hash_val in hashes)
                if all_same:
                    print("\n>>All local global model hashes are the same. Proceeding to the next round.")
                    # ========== 启动下一轮 ==========
                    print("\n>>Central Server Starting Next Round...")
                    tx_hash = central_server_client.sendRawTransaction(contract_address, contract_abi, "startNextRound",)
                    print(f"Central Server sent transaction to start Round {args.current_round + 1}. Transaction hash: {tx_hash}")
                    receipt = central_server_client.getTransactionReceipt(tx_hash)
                    print(f"Central Server started Round {args.current_round + 1}. Receipt: {receipt}")
                    if not (receipt and receipt['status'] == 0):
                        print(f"Error starting next round. Receipt: {receipt}")
                        return False
                    log_operation(log_dir, args.current_round, "server", "next_round_started", f"Central Server started Round {args.current_round + 1}.")
                    return True
                else:
                    print("\n>>Local global model hashes are different. Stopping the process.")
                    log_operation(log_dir, args.current_round, "server", "hash_check_fail", f"Local global model hashes are different in Round {args.current_round}. Stopping.")
                    return False
            else:
                print("\n>>No hashes received from participants.")
                log_operation(log_dir, args.current_round, "server", "no_hashes_received", f"No hashes received from participants in Round {args.current_round}.")
                return False
        except Exception as e:
            print(f"\n>>Error retrieving or comparing hashes: {e}")
            log_operation(log_dir, args.current_round, "server", "hash_check_error", f"Error retrieving or comparing hashes in Round {args.current_round}: {e}")
            return False
        finally:
            log_operation(log_dir, args.current_round, "server", "hash_check_complete", f"Central Server Hash Checking Completed for Round {args.current_round}.")
            print(f"\n>>Central Server (Hash Checking and Next Round) Completed Round: {args.current_round}")

    else:
        raise ValueError(f"Invalid run_mode value: {run_mode}. Must be 0, 1, 2, or 3.")
    return True

# ==========  参与者节点角色函数 (多节点版本 -  训练和上传) ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, participant_id="participant1", log_dir="fl_multi_log"):
    print(f"\n>>Starting Participant Node ({participant_id} - Decentralized Aggregation)...")
    log_operation(log_dir, args.current_round, participant_id, "training_node_start", f"Participant Node {participant_id} Training Node started (Decentralized Aggregation).")

    # ----- 1. 下载更新信息（包含参与者ID和模型更新） -----
    print(f"\n>>Participant Node ({participant_id}): Downloading updates from other participants (Round {args.currentRound})...")
    log_operation(log_dir, args.current_round, participant_id, "download_other_updates_start", f"Participant Node {participant_id} starts downloading updates from others.")
    participant_updates = {}
    try:
        participant_updates_json_str = participant_client.call(contract_address, contract_abi, "getParticipantUpdates", [args.current_round])[0]
        updates_json = json.loads(participant_updates_json_str)
        for update_info in updates_json:
            other_participant_id = update_info['participantId']
            model_update_str = update_info['modelUpdate']
            if other_participant_id != participant_id:  # Don't download own update
                participant_updates[other_participant_id] = deserialize_model(model_update_str, model_type=args.model)
        print(f"\n>>Participant Node ({participant_id}): Downloaded updates from {len(participant_updates)} other participants.")
        log_operation(log_dir, args.current_round, participant_id, "download_other_updates_success", f"Participant Node {participant_id} downloaded updates from others successfully.")
    except Exception as e:
        print(f"\n>>Participant Node ({participant_id}): Error downloading updates from others: {e}")
        log_operation(log_dir, args.current_round, participant_id, "download_other_updates_fail", f"Participant Node {participant_id} failed to download updates from others: {e}")
        return

    # ----- 2. 本地模型训练 (获取本轮自己的更新) -----
    print(f"\n>>Participant Node ({participant_id}): Loading Training Data for Round {args.currentRound}...")
    if args.dataset == 'cifar10':
        train_loader = load_cifar10_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants)
    else:
        train_loader = load_mnist_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants)
    print(f"\n>>Participant Node ({participant_id}): Starting Local Model Training for Round {args.currentRound}...")
    log_operation(log_dir, args.current_round, participant_id, "local_training_start", f"Participant Node {participant_id} local model training started for Round {args.currentRound}. ")
    local_model = MLP() if args.model == 'mlp' else CNN() # 为本轮训练初始化一个新的本地模型
    trained_model = train_model(local_model, train_loader, epochs=args.epochs, log_dir=log_dir, round_num=args.current_round, participant_id=participant_id)
    own_update_str = serialize_model(trained_model)
    upload_model_update(participant_client, contract_abi, contract_address, own_update_str, participant_id, args.current_round) # 上传自己的更新

    # ----- 3. 本地聚合 -----
    print(f"\n>>Participant Node ({participant_id}): Performing local aggregation...")
    log_operation(log_dir, args.current_round, participant_id, "local_aggregation_start", f"Participant Node {participant_id} starts local aggregation.")
    aggregated_model = aggregate_global_model_decentralized(trained_model, participant_updates) # 使用本轮训练的模型和上一轮其他参与者的更新进行聚合
    aggregated_model_str = serialize_model(aggregated_model)
    log_operation(log_dir, args.current_round, participant_id, "local_aggregation_success", f"Participant Node {participant_id} local aggregation finished.")

    # ----- 4. 计算并提交本地聚合模型的哈希 -----
    print(f"\n>>Participant Node ({participant_id}): Submitting hash of locally aggregated model...")
    log_operation(log_dir, args.current_round, participant_id, "submit_hash_start", f"Participant Node {participant_id} starts submitting hash of aggregated model.")
    local_global_model_hash = torch.sha256(torch.tensor(aggregated_model_str.encode('utf-8')))
    submit_local_global_model_hash(participant_client, contract_abi, contract_address, local_global_model_hash.hex(), participant_id, args.current_round)

    # ----- 5. 本地评估聚合后的模型 -----
    print(f"\n>>Participant Node ({participant_id}): Evaluating locally aggregated model...")
    log_operation(log_dir, args.current_round, participant_id, "local_evaluation_start", f"Participant Node {participant_id} starts evaluating locally aggregated model.")
    if args.dataset == 'cifar10':
        test_loader = load_cifar10_test_data()
    else:
        test_loader = load_mnist_test_data()
    evaluate_model(aggregated_model, test_loader, log_dir=log_dir, round_num=args.current_round, participant_id=participant_id + "_local_agg") # 区分评估日志
    log_operation(log_dir, args.current_round, participant_id, "local_evaluation_success", f"Participant Node {participant_id} local evaluation finished.")

    log_operation(log_dir, args.current_round, participant_id, "training_node_complete", f"Participant Node {participant_id} Training Node Finished Round: {args.currentRound} (Decentralized Aggregation).")
    print(f"\n>>Participant Node ({participant_id} - Decentralized Aggregation) Finished Round: {args.currentRound}")
    
# ==========  主程序入口 (Demo 模式 -  多进程多客户端，多节点模拟) ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Round Training, Decentralized Aggregation, Hash Verification)")


    # ==========  添加参数解析 ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script (Multi-Node Version)")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round')
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds')
    parser.add_argument('--log_dir', type=str, default='fl_multi_log', help='Directory to save log files')
    parser.add_argument('--num_participants', type=int, default=3, help='Number of participant nodes to simulate')
    parser.add_argument('--address', type=str, default='', help='Contract address to interact with')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for hash comparison (if applicable)')
    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Log Dir: {args.log_dir}, Num Participants: {args.num_participants}, Window Size: {args.window_size}")

    log_dir = args.log_dir
    num_participants = args.num_participants

    # ========== 清空日志文件夹 ==========
    if os.path.exists(log_dir):
        print(f"\n>>Clearing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    central_server_client = Bcos3Client()
    participant_clients = []
    for i in range(num_participants):
        client_config=ClientConfig()
        client_config.account_keyfile = f"client{i}.keystore"
        client_config.account_password = f"{i}"*6
        participant_clients.append(Bcos3Client(client_config))


    # ========== 运行中央服务器进行部署和设置 ==========
    args.current_round = 0
    run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=0, participant_clients=participant_clients, log_dir=log_dir)

    # ========== 运行中央服务器进行参与者注册 ==========
    args.current_round = 0
    run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=2, participant_clients=participant_clients, log_dir=log_dir)

    # ========== 运行中央服务器的准备阶段 ==========
    args.current_round = 0
    run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=1, log_dir=log_dir)


    # ==========  多轮联邦学习循环  ==========
    print(f"\n>>[DEBUG - Main Loop Start] Rounds: {args.rounds}")
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}")
        args.current_round = round_num
        args.num_participants = num_participants - 1 if num_participants > 0 else 0

        # ========== 模拟参与者节点训练和本地聚合，并提交哈希 ==========
        print(f"\n>>Round {round_num} - Simulating Participant Nodes (Training, Decentralized Aggregation, and Hash Submission)...")
        participant_ids = [f"participant{i}" for i in range(1, num_participants)]
        for i, participant_id in enumerate(participant_ids):
            print(f"\n>>Starting Participant Node ({participant_id})...")
            run_participant_node(participant_clients[i+1], contract_abi, CONTRACT_ADDRESS, args, participant_id=participant_id, log_dir=log_dir)

        # ========== 中心节点获取哈希并检查，启动下一轮 ==========
        args.current_round = round_num
        if not run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=3, log_dir=log_dir):
            break

        # ========== 模型评估 (评估在每个参与者本地进行) ==========
        print(f"\n>>Round {round_num} - Central Server (Skipping Evaluation - Performed Locally by Participants)...")

    print("\n>>Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Round Training, Decentralized Aggregation, Hash Verification) Finished!")