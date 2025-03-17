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
from myclient.fl_multi_utils import ( #  !!!  导入 fl_multi_utils  !!!
    log_operation,
    load_mnist_data_partition_multi, #  !!!  导入 multi 版本的数据加载函数  !!!
    load_mnist_test_data,
    load_cifar10_data_partition_multi, #  !!!  导入 multi 版本的数据加载函数  !!!
    load_cifar10_test_data,
    MLP,
    CNN,
    serialize_model,
    deserialize_model,
    train_model,
    evaluate_model,
    upload_model_update,
    download_global_model,
    aggregate_global_model, #  !!!  聚合函数暂时不导入，后面再添加 !!!
    upload_global_model,    #  !!!  上传全局模型函数暂时不导入，后面再添加 !!!
)

# ----- 合约信息 (需要根据你的实际情况修改) -----
CONTRACT_NAME = "EnhancedFederatedLearning" #  !!!  合约名称保持不变，仍然使用 EnhancedFederatedLearning !!!
CONTRACT_PATH = "./contracts/EnhancedFederatedLearning.sol" #  !!!  合约路径保持不变 !!!
ABI_PATH = "./contracts/EnhancedFederatedLearning.abi" #  !!!  ABI 路径保持不变 !!!
BIN_PATH = "./contracts/EnhancedFederatedLearning.bin" #  !!!  BIN 路径保持不变 !!!
CONTRACT_NOTE_NAME = "federatedlearning_multi_demo" #  !!!  修改 Contract Note 名称为 multi 版本 !!!
CONTRACT_ADDRESS = "" #  !!!  合约地址硬编码在这里 !!!

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
def run_central_server(central_server_client, contract_abi, contract_address, args, run_mode, log_dir="fl_multi_log"): #  !!!  修改默认 log_dir !!!
    if run_mode == 1: 
        print(f"\n>>Starting Central Server (Preparation Node)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Preparation) <<<<<<")
        print(f"\n>>Central Server (Preparation Node) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "preparation_start", "Central Server Preparation Node started.")
    elif run_mode == 2:
        print(f"\n>>Starting Central Server (Aggregation Node)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Aggregation) <<<<<<")
        print(f"\n>>Central Server (Aggregation Node) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "aggregation_start", "Central Server Aggregation Node started.")
    elif run_mode == 3: 
        print(f"\n>>Starting Central Server (Testing Node - Evaluation)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Evaluation) <<<<<<")
        print(f"\n>>Central Server (Testing Node - Evaluation) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "evaluation_start", "Central Server Evaluation Node started.")
    else:
        raise ValueError(f"Invalid run_mode value: {run_mode}. Must be 1, 2 or 3.")


    # -----  1. 模型下载 (Aggregation, Evaluation Node 都需要下载全局模型) -----
    if run_mode in [2, 3]:
        print("\n>>Central Server: Downloading Global Model...")
        log_operation(log_dir, args.current_round, "server", "download_global_model_start", "Central Server starts downloading global model.")
        downloaded_model_str = download_global_model(central_server_client, contract_abi, contract_address, None, "server")

        if downloaded_model_str:
            print("\n>>Global Model downloaded successfully.")
            log_operation(log_dir, args.current_round, "server", "download_global_model_success", "Global model downloaded successfully.")
            global_model = deserialize_model(downloaded_model_str, model_type=args.model)
        else:
            print("\n>>Failed to download Global Model.")
            log_operation(log_dir, args.current_round, "server", "download_global_model_fail", "Failed to download global model.")
            if run_mode == 2:
                print("\n>>Aggregation Failed: No Global Model Downloaded for Aggregation.")
                log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to global model download failure.")
                return
            elif run_mode == 3:
                print("\n>>Evaluation Failed: No Global Model Downloaded for Evaluation.")
                log_operation(log_dir, args.current_round, "server", "evaluation_fail", "Evaluation failed due to global model download failure.")
                return


    if run_mode == 2: 
        # ----- 2. 下载参与者模型更新 -----
        print("\n>>Central Server (Aggregation Node): Downloading Participant Model Updates...")
        log_operation(log_dir, args.current_round, "server", "download_participant_updates_start", "Central Server starts downloading participant model updates.")
        participant_updates_json_str = central_server_client.call(contract_address, contract_abi, "getParticipantUpdates", [args.current_round])[0] #  !!!  关键点:  使用 [0] 获取返回值列表的第一个元素 !!!
        if participant_updates_json_str:
            print("\n>>Participant Model Updates downloaded successfully (JSON String).")
            log_operation(log_dir, args.current_round, "server", "download_participant_updates_success", "Participant model updates downloaded successfully.")
            participant_updates = json.loads(participant_updates_json_str) #  !!!  JSON 解析 !!!
            print(f"\n>>[DEBUG - Aggregation Node] Participant Updates (JSON): \n{json.dumps(participant_updates, indent=2)}") # DEBUG - 打印 JSON 数据
        else:
            print("\n>>Failed to download Participant Model Updates.")
            log_operation(log_dir, args.current_round, "server", "download_participant_updates_fail", "Failed to download participant model updates.")
            print("\n>>Aggregation Failed: No Participant Model Updates Downloaded.")
            log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to participant model updates download failure.")
            return

        # ----- 3. 模型聚合 (简单的平均聚合) -----
        print("\n>>Central Server (Aggregation Node): Aggregating Model Updates...")
        log_operation(log_dir, args.current_round, "server", "aggregation_model_start", "Central Server starts aggregating model updates.")
        aggregated_model = aggregate_global_model(global_model, participant_updates, model_type=args.model) #  !!!  聚合函数还没定义， 后面再添加 !!!
        aggregated_model_str = serialize_model(aggregated_model)
        log_operation(log_dir, args.current_round, "server", "aggregation_model_success", "Global model aggregation finished.")

        # ----- 4. 上传聚合后的全局模型 -----
        print("\n>>Central Server (Aggregation Node): Uploading Aggregated Global Model...")
        log_operation(log_dir, args.current_round, "server", "upload_aggregated_model_start", "Central Server starts uploading aggregated global model.")
        if upload_global_model(central_server_client, contract_abi, contract_address, aggregated_model_str): #  !!! 上传全局模型函数还没定义， 后面再添加 !!!
            print("\n>>Aggregated Global Model uploaded successfully.")
            log_operation(log_dir, args.current_round, "server", "upload_aggregated_model_success", "Aggregated global model uploaded successfully.")
        else:
            print("\n>>Failed to upload Aggregated Global Model.")
            log_operation(log_dir, args.current_round, "server", "upload_aggregated_model_fail", "Failed to upload aggregated global model.")
            print("\n>>Aggregation Failed: Failed to Upload Aggregated Global Model.")
            log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to aggregated global model upload failure.")
            return
        log_operation(log_dir, args.current_round, "server", "aggregation_complete", f"Central Server Aggregation Node Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Aggregation Node) Completed Round: {args.current_round}")


    elif run_mode == 3: 
        # ----- 2. 加载测试数据集 -----
        print("\n>>Central Server (Testing Node - Evaluation): Loading Test Data...")
        if args.dataset == 'cifar10':
            test_loader = load_cifar10_test_data()
        else:
            test_loader = load_mnist_test_data()

        # ----- 3. 模型评估 -----
        print("\n>>Central Server (Testing Node - Evaluation): Evaluating Global Model...")
        log_operation(log_dir, args.current_round, "server", "evaluation_model_start", "Central Server starts evaluating global model.")
        evaluate_model(global_model, test_loader, log_dir=log_dir, round_num=args.current_round)
        log_operation(log_dir, args.current_round, "server", "evaluation_model_success", "Global model evaluation finished.")
        log_operation(log_dir, args.current_round, "server", "evaluation_complete", f"Central Server Evaluation Node Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Testing Node - Evaluation) Completed Round: {args.current_round}")


    elif run_mode == 1: # 第一次运行 (Preparation Node)
        log_operation(log_dir, args.current_round, "server", "preparation_complete", f"Central Server Preparation Node Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Preparation Node) Completed Round: {args.current_round}")



# ==========  参与者节点角色函数 (多节点版本 -  训练和上传) ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, participant_id="participant1", log_dir="fl_multi_log"): #  !!!  修改默认 log_dir !!!
    # ... (函数开始部分，打印信息，与双节点版本类似，需要修改 participant_id 的打印) ...
    log_operation(log_dir, args.current_round, participant_id, "training_node_start", f"Participant Node {participant_id} Training Node started.")

    # ----- 1. 下载全局模型 -----
    print(f"\n>>Participant Node ({participant_id}): Downloading Global Model...")
    log_operation(log_dir, args.current_round, participant_id, "download_global_model_start", f"Participant Node {participant_id} starts downloading global model.")
    downloaded_model_str = download_global_model(participant_client, contract_abi, contract_address, None, participant_id) #  !!!  角色名使用 participant_id !!!
    print(f"\n>>[DEBUG - Participant Node] Downloaded Model String: {downloaded_model_str}") # DEBUG - Print downloaded string # ADDED DEBUG PRINT
    if downloaded_model_str:
        print(f"\n>>Global Model downloaded successfully (for Participant: {participant_id}).")
        log_operation(log_dir, args.current_round, participant_id, "download_global_model_success", f"Participant Node {participant_id} global model downloaded successfully.")
        model = deserialize_model(downloaded_model_str, model_type=args.model)
    else:
        print(f"\n>>Failed to download Global Model (for Participant: {participant_id}). Using local initial model.")
        log_operation(log_dir, args.current_round, participant_id, "download_global_model_fail", f"Participant Node {participant_id} failed to download global model, using local initial model.")
        if args.model == 'cnn':
            model = CNN()
        else:
            model = MLP()

    # ----- 2. 本地模型训练 -----
    print(f"\n>>Participant Node ({participant_id}): Loading Training Data...")
    if args.dataset == 'cifar10':
        train_loader = load_cifar10_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants) #  !!!  加载 multi 版本数据, 传递 participant_id 和 total_participants !!!
    else:
        train_loader = load_mnist_data_partition_multi(participant_id=participant_id, total_participants=args.num_participants) #  !!!  加载 multi 版本数据, 传递 participant_id 和 total_participants !!!
    print(f"\n>>Participant Node ({participant_id}): Starting Local Model Training...")
    log_operation(log_dir, args.current_round, participant_id, "local_training_start", f"Participant Node {participant_id} local model training started.")
    trained_model = train_model(model, train_loader, epochs=args.epochs, log_dir=log_dir, round_num=args.current_round, participant_id=participant_id) #  !!!  传递 participant_id !!!
    updated_model_str = serialize_model(trained_model)
    log_operation(log_dir, args.current_round, participant_id, "local_training_success", f"Participant Node {participant_id} local model training finished.")

    # ----- 3. 上传模型更新 -----
    print(f"\n>>Participant Node ({participant_id}): Uploading Model Update...")
    log_operation(log_dir, args.current_round, participant_id, "upload_model_update_start", f"Participant Node {participant_id} starts uploading model update.")
    if upload_model_update(participant_client, contract_abi, contract_address, updated_model_str, participant_id, args.current_round): #  !!!  角色名使用 participant_id, 传递 round_num !!!
        print(f"\n>>Model update uploaded successfully (from Participant: {participant_id}).")
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_success", f"Participant Node {participant_id} model update uploaded successfully.")
    else:
        print(f"\n>>Model update upload failed (from Participant: {participant_id}).")
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_fail", f"Participant Node {participant_id} model update upload failed.")

    log_operation(log_dir, args.current_round, participant_id, "training_node_complete", f"Participant Node {participant_id} Training Node Finished Round: {args.current_round}")
    print(f"\n>>Participant Node ({participant_id} - Training Node) Finished Round: {args.current_round}")



# ==========  主程序入口 (Demo 模式 -  多进程多客户端，多节点模拟) ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server):----------------------------------------------------------") #  !!! 修改打印信息 -  标记为 Multi-Node Version

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Multi-Client Simulation - Single Process, Multi-Process Structure, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server - Multi-Node Version)") #  !!! 修改打印信息，更清晰地表达当前模式 -  Multi-Node Version


    # ==========  添加参数解析 (复用 federatedlearning_single.py 的参数,  添加 num_participants 参数) ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script (Multi-Node Version)") #  !!! 修改 description - Multi-Node Version !!!
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use (mlp or cnn)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round (default: 1)')
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds (default: 2)')
    parser.add_argument('--role', type=str, default='demo', choices=['demo', 'server', 'participant'], help='Role to run (demo, server, participant)') # 角色参数 (虽然 demo 模式固定为 demo)
    parser.add_argument('--log_dir', type=str, default='fl_multi_log', help='Directory to save log files (default: fl_multi_log)') #  !!!  日志文件夹参数修改为 fl_multi_log !!!
    parser.add_argument('--num_participants', type=int, default=3, help='Number of participant nodes to simulate in demo mode (default: 3)') #  !!!  新增 num_participants 参数 !!!
    parser.add_argument('--address', type=str, default='', help='Contract address to interact with') #  !!!  移除 address 参数 !!!
    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Role: {args.role}, Log Dir: {args.log_dir}, Num Participants: {args.num_participants}") #  !!!  打印配置信息 -  去除 Contract Address, 新增 Num Participants !!!


    role = args.role #  从命令行参数获取角色 (虽然 demo 模式固定为 demo)
    log_dir = args.log_dir #  从命令行参数获取日志文件夹路径
    num_participants = args.num_participants #  从命令行参数获取参与者数量


    # ========== 清空日志文件夹 (如果存在) ==========
    if os.path.exists(log_dir):
        print(f"\n>>Clearing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    else:
        print(f"\n>>Log directory not found, creating: {log_dir}")

    # ========== 创建日志文件夹 ==========
    os.makedirs(log_dir, exist_ok=True)


    central_server_client = Bcos3Client() #  创建中央服务器客户端实例
    participant_clients = [] #  !!!  使用列表存储多个参与者客户端实例 !!!
    for i in range(num_participants): #  !!!  根据 num_participants 创建多个参与者客户端 !!!
        client_config=ClientConfig()
        client_config.account_keyfile = f"client{i}.keystore" #!!!  根据参与者 ID 生成 keystore 文件名!!!
        client_config.account_password = f"{i}"*6 #!!!  固定密码!!!
        participant_clients.append(Bcos3Client(client_config)) #  为每个参与者创建一个客户端实例


    # ========== 部署合约 (Demo Mode - Server Part - Preparation) - 部署合约只在第一轮之前进行 ==========
    CONTRACT_ADDRESS = args.address # 从命令行参数获取合约地址
    if not CONTRACT_ADDRESS: # 使用硬编码的 CONTRACT_ADDRESS，这里判断可以移除，直接部署
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Deploying contract...")
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = central_server_client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"] # 更新 CONTRACT_ADDRESS 为实际部署地址
        with open("fl_multi_log.txt", 'a') as f:
            f.write(f"\n[Parameters @ {datetime.datetime.now()}]\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Rounds: {args.rounds}\n") 
            f.write(f"Participants: {args.num_participants}\n")
            f.write(f"Contract Address: {CONTRACT_ADDRESS}\n")
            f.write("-"*40 + "\n")
        print(f"Demo Mode - Central Server (Preparation Node): Deploy contract success, contract address: {CONTRACT_ADDRESS}")
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
    else: # 即使有硬编码地址，也提示正在使用
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Using existing contract at hardcoded address: {CONTRACT_ADDRESS}")


    # ==========  多轮联邦学习循环  ==========
    print(f"\n>>[DEBUG - Main Loop Start] Rounds: {args.rounds}") # DEBUG - 输出总轮数
    for round_num in range(1, args.rounds + 1): # 从第 1 轮循环到 rounds 轮
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}") # 添加轮次分隔符
        args.current_round = round_num # 将当前轮数添加到 args 中
        args.num_participants = num_participants #  !!!  将参与者数量添加到 args 中 !!!
        print(f"\n>>[DEBUG - Main Loop - Round Start] Current Round: {args.current_round}, Num Participants: {args.num_participants}") # DEBUG - 输出每轮循环开始时的轮数


        # ========== 每一轮的第一次运行中央服务器角色逻辑 (Demo Mode - Server Part - Preparation) ==========
        print(f"\n>>Round {round_num} - First Run: Central Server (Preparation Node) - Model Preparation...")
        print(f"\n>>[DEBUG - Central Server Preparation Start] Round: {args.current_round}, run_mode: 1")
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=1, log_dir=log_dir) # 顺序执行中央服务器逻辑 (Preparation), 传递 args, run_mode=1, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Preparation End] Round: {args.current_round}, run_mode: 1")
        print(f"\n>>Round {round_num} - First Run: Demo Mode - Central Server (Preparation Node) process finished.")


        # ========== 模拟参与者节点角色 (Demo Mode - Participant Part - Training and Upload) - 模拟多个参与者节点并行训练 ==========
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training Nodes) process starting...") # 修改打印信息 - 复数 Nodes
        print(f"\n>>[DEBUG - Participant Nodes Start] Round: {args.current_round}") # DEBUG - 输出 Participant Nodes 开始时的轮数

        # -----  模拟多个参与者节点并行训练 -----
        participant_ids = [f"participant{i+1}" for i in range(num_participants)] #  !!!  生成参与者 ID 列表 (participant1, participant2, participant3...) !!!
        for i, participant_id in enumerate(participant_ids): #  !!!  循环启动多个参与者节点 !!!
            print(f"\n>>Starting Participant Node ({participant_id})...")
            run_participant_node(participant_clients[i], contract_abi, CONTRACT_ADDRESS, args, participant_id=participant_id, log_dir=log_dir) #  !!!  为每个参与者传递不同的 participant_id 和 client 实例 !!!

        print(f"\n>>[DEBUG - Participant Nodes End] Round: {args.current_round}") # DEBUG - 输出 Participant Nodes 结束时的轮数
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training Nodes) process finished.") # 修改打印信息 - 复数 Nodes


        # ========== 每一轮的第二次运行中央服务器角色逻辑 (Demo Mode - Server Part - Aggregation) ==========
        print(f"\n>>Round {round_num} - Second Run: Central Server (Aggregation Node) - Model Aggregation...")
        print(f"\n>>[DEBUG - Central Server Aggregation Start] Round: {args.current_round}, run_mode: 2")
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=2, log_dir=log_dir) # 顺序执行中央服务器逻辑 (Aggregation), 传递 args, run_mode=2, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Aggregation End] Round: {args.current_round}, run_mode: 2")
        print(f"\n>>Round {round_num} - Second Run: Demo Mode - Central Server (Aggregation Node) process finished.")


        # ========== 每一轮的第三次运行中央服务器角色逻辑 (Demo Mode - Server Part - Evaluation) - 每轮训练后都评估 ==========
        print(f"\n>>Round {round_num} - Third Run: Central Server (Testing Node - Evaluation) - Model Evaluation after Round {round_num} Aggregation...")
        print(f"\n>>[DEBUG - Central Server Evaluation Start] Round: {args.current_round}, run_mode: 3")
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=3, log_dir=log_dir) # 每轮训练后都进行模型评估, 传递 args, run_mode=3, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Evaluation End] Round: {args.current_round}, run_mode: 3")
        print(f"\n>>Round {round_num} - Third Run: Demo Mode - Central Server (Testing Node - Evaluation) process finished.")
        print(f"\n>>[DEBUG - Main Loop - Round End] Current Round: {args.current_round}") # DEBUG - 输出每轮循环结束时的轮数


    print("\n>>Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server) Finished!") # 修改打印信息，更清晰地表达当前模式