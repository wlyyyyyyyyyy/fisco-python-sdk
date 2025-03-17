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
from client_config import client_config as ClientConfig
import multiprocessing #  !!!  导入 multiprocessing 模块 !!!
import time #  !!!  导入 time 模块， 用于 sleep 和超时控制 !!!

# =====  导入 fl_multi_utils.py 中的通用函数和类  =====
from myclient.flp_multi_utils import ( #  !!!  导入 fl_multi_utils  !!!
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
    aggregate_global_model,
    upload_global_model,
)

# ----- 合约信息 (需要根据你的实际情况修改) -----
CONTRACT_NAME = "EnhancedFederatedLearning" #  !!!  合约名称保持不变，仍然使用 EnhancedFederatedLearning !!!
CONTRACT_PATH = "./contracts/EnhancedFederatedLearning.sol" #  !!!  合约路径保持不变 !!!
ABI_PATH = "./contracts/EnhancedFederatedLearning.abi" #  !!!  ABI 路径保持不变 !!!
BIN_PATH = "./contracts/EnhancedFederatedLearning.bin" #  !!!  BIN 路径保持不变 !!!
CONTRACT_NOTE_NAME = "federatedlearning_multi_demo" #  !!!  修改 Contract Note 名称为 multi 版本 !!!
CONTRACT_ADDRESS = "" #  合约地址可以从命令行参数传入

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
    try: #  !!!  添加 try-except 块， 捕获异常 !!!
        if run_mode == 1: # 第一次运行 (Preparation Node)
            print(f"\n>>Starting Central Server (Preparation Node)...")
            print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Preparation) <<<<<<")
            print(f"\n>>Central Server (Preparation Node) in Round: {args.current_round}")
            log_operation(log_dir, args.current_round, "server", "preparation_start", "Central Server Preparation Node started.")
        elif run_mode == 2: # 第二次运行 (Aggregation Node)
            print(f"\n>>Starting Central Server (Aggregation Node)...")
            print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Aggregation) <<<<<<")
            print(f"\n>>Central Server (Aggregation Node) in Round: {args.current_round}")
            log_operation(log_dir, args.current_round, "server", "aggregation_start", "Central Server Aggregation Node started.")
        elif run_mode == 3: # 第三次运行 (Testing/Evaluation Node)
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


        if run_mode == 2: # 第二次运行 (Aggregation Node) - 模型聚合和更新
            # ----- 2. 下载参与者模型更新 -----
            print("\n>>Central Server (Aggregation Node): Downloading Participant Model Updates...")
            log_operation(log_dir, args.current_round, "server", "download_participant_updates_start", "Central Server starts downloading participant model updates.")
            participant_updates_json_str = central_server_client.call(contract_address, contract_abi, "getParticipantUpdates", [args.current_round])[0] #  !!!  关键点:  使用 [0] 获取返回值列表的第一个元素 !!!
            if participant_updates_json_str:
                print("\n>>Participant Model Updates downloaded successfully (JSON String).")
                log_operation(log_dir, args.current_round, "server", "download_participant_updates_success", "Participant model updates downloaded successfully.")
                try: # !!! 添加 try-except 块， 捕获 JSON 解析异常 !!!
                    participant_updates = json.loads(participant_updates_json_str) #  !!!  JSON 解析 !!!
                    print(f"\n>>[DEBUG - Aggregation Node] Participant Updates (JSON): \n{json.dumps(participant_updates, indent=2)}") # DEBUG - 打印 JSON 数据
                except json.JSONDecodeError as e: # !!! 捕获 JSON 解析异常 !!!
                    error_message = f"JSONDecodeError: {e}, JSON String: {participant_updates_json_str}"
                    print(f"\n>>[ERROR] {error_message}")
                    log_operation(log_dir, args.current_round, "server", "json_decode_error", error_message)
                    log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to JSON decode error.")
                    return #  !!!  JSON 解析失败， 结束 Aggregation 节点 !!!

            else:
                print("\n>>Failed to download Participant Model Updates.")
                log_operation(log_dir, args.current_round, "server", "download_participant_updates_fail", "Failed to download participant model updates.")
                print("\n>>Aggregation Failed: No Participant Model Updates Downloaded.")
                log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to participant model updates download failure.")
                return

            # ----- 3. 模型聚合 (简单的平均聚合) -----
            print("\n>>Central Server (Aggregation Node): Aggregating Model Updates...")
            log_operation(log_dir, args.current_round, "server", "aggregation_model_start", "Central Server starts aggregating model updates.")
            aggregated_model = aggregate_global_model(global_model, participant_updates, model_type=args.model) #  !!!  聚合函数 !!!
            aggregated_model_str = serialize_model(aggregated_model)
            log_operation(log_dir, args.current_round, "server", "aggregation_model_success", "Global model aggregation finished.")

            # ----- 4. 上传聚合后的全局模型 -----
            print("\n>>Central Server (Aggregation Node): Uploading Aggregated Global Model...")
            log_operation(log_dir, args.current_round, "server", "upload_aggregated_model_start", "Central Server starts uploading aggregated global model.")
            if upload_global_model(central_server_client, contract_abi, contract_address, aggregated_model_str): #  !!! 上传全局模型函数 !!!
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


        elif run_mode == 3: # 第三次运行 (Testing Node - Evaluation) - 模型评估节点
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

    except Exception as e: #  !!!  捕获所有异常 !!!
        error_message = f"Exception in Central Server Node (Run Mode: {run_mode}), Round: {args.current_round}. Error: {e}, Traceback: {traceback.format_exc()}"
        print(f"\n>>[ERROR] {error_message}")
        log_operation(log_dir, args.current_round, "server", f"server_node_exception_run_mode_{run_mode}", error_message)
        print(f"\n>>Central Server Node (Run Mode: {run_mode}) Failed in Round {args.current_round}. See error log for details.")


# ==========  参与者节点角色函数 (多节点版本 -  训练和上传) ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, participant_id="participant1", log_dir="fl_multi_log"): #  !!!  修改默认 log_dir !!!
    try: #  !!!  添加 try-except 块, 捕获异常 !!!
        print(f"\n>>Starting Participant Node (Training Node), Participant ID: {participant_id}...") #  !!! 修改打印信息， 包含 participant_id !!!
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Participant Node - {participant_id}) <<<<<<") #  !!! 修改打印信息， 包含 participant_id !!!
        print(f"\n>>Participant Node ({participant_id} - Training Node) in Round: {args.current_round}") #  !!! 修改打印信息， 包含 participant_id !!!
        # ... (函数开始部分，打印信息，与双节点版本类似，需要修改 participant_id 的打印) ...
        log_operation(log_dir, args.current_round, participant_id, "training_node_start", f"Participant Node {participant_id} Training Node started.")

        # ----- 1. 下载全局模型 -----
        print(f"\n>>Participant Node ({participant_id}): Downloading Global Model...")
        log_operation(log_dir, args.current_round, participant_id, "download_global_model_start", f"Participant Node {participant_id} starts downloading global model.")
        downloaded_model_str = download_global_model(participant_client, contract_abi, contract_address, None, participant_id) #  !!!  角色名使用 participant_id !!!
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

    except Exception as e: #  !!!  捕获所有异常 !!!
        error_message = f"Exception in Participant Node (Process ID: {os.getpid()}), Participant ID: {participant_id}, Round: {args.current_round}. Error: {e}, Traceback: {traceback.format_exc()}" #  !!!  添加进程 ID 到错误信息 !!!
        print(f"\n>>[ERROR] {error_message}")
        log_operation(log_dir, args.current_round, participant_id, "participant_node_exception", error_message)
        print(f"\n>>Participant Node ({participant_id}) Training Failed in Round {args.current_round} (Process ID: {os.getpid()}). See error log for details.") #  !!!  添加进程 ID 到错误信息 !!!


# ==========  主程序入口 (Demo 模式 -  多进程多客户端，多节点模拟) ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Process):----------------------------------------------------------") #  !!! 修改打印信息 -  标记为 Multi-Process

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Multi-Client Simulation - Multi-Process Structure, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server - Multi-Node Version)") #  !!! 修改打印信息，更清晰地表达当前模式 -  Multi-Process

    # ==========  添加参数解析 (复用 federatedlearning_single.py 的参数,  添加 num_participants 参数) ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script (Multi-Process Multi-Node Version)") #  !!! 修改 description - Multi-Process Multi-Node Version !!!
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use (mlp or cnn)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round (default: 1)')
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds (default: 2)')
    parser.add_argument('--contract_address', type=str, default='', help='Optional: Contract address to use. If not provided, a new contract will be deployed.')
    parser.add_argument('--role', type=str, default='demo', choices=['demo', 'server', 'participant'], help='Role to run (demo, server, participant)') # 角色参数 (虽然 demo 模式固定为 demo)
    parser.add_argument('--log_dir', type=str, default='fl_multi_log', help='Directory to save log files (default: fl_multi_log)') #  !!!  日志文件夹参数修改为 fl_multi_log !!!
    parser.add_argument('--num_participants', type=int, default=3, help='Number of participant nodes to simulate in demo mode (default: 3)') #  !!!  新增 num_participants 参数 !!!
    parser.add_argument('--wait_time', type=int, default=2, help='Wait time after each round in demo mode (default: 2 seconds)') #  !!! 新增 wait_time 参数 !!!


    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Role: {args.role}, Contract Address: {args.contract_address}, Log Dir: {args.log_dir}, Num Participants: {args.num_participants}, Wait Time: {args.wait_time} seconds") #  !!!  打印配置信息 -  新增 Num Participants 和 Wait Time !!!


    CONTRACT_ADDRESS = args.contract_address
    role = args.role #  从命令行参数获取角色 (虽然 demo 模式固定为 demo)
    log_dir = args.log_dir #  从命令行参数获取日志文件夹路径
    num_participants = args.num_participants #  从命令行参数获取参与者数量
    wait_time = args.wait_time #  从命令行参数获取轮次等待时间


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
    if not CONTRACT_ADDRESS:
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Deploying contract...")
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = central_server_client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"]
        print(f"Demo Mode - Central Server (Preparation Node): Deploy contract success, contract address: {CONTRACT_ADDRESS}")
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
    else:
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Using existing contract at: {CONTRACT_ADDRESS}")


    # ==========  多轮联邦学习循环  ==========
    print(f"\n>>[DEBUG - Main Loop Start] Rounds: {args.rounds}") # DEBUG - 输出总轮数
    for round_num in range(1, args.rounds + 1): # 从第 1 轮循环到 rounds 轮
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}") # 添加轮次分隔符
        args.current_round = round_num # 将当前轮数添加到 args 中
        args.num_participants = num_participants #  !!!  将参与者数量添加到 args 中 !!!
        print(f"\n>>[DEBUG - Main Loop - Round Start] Current Round: {args.current_round}, Num Participants: {args.num_participants}") # DEBUG - 输出每轮循环开始时的轮数


        # ========== 每一轮的第一次运行中央服务器角色逻辑 (Demo Mode - Server Part - Preparation) - 多进程 ==========
        print(f"\n>>Round {round_num} - First Run: Starting Central Server (Preparation Node) Process...")
        preparation_process = multiprocessing.Process(target=run_central_server, args=(central_server_client, contract_abi, CONTRACT_ADDRESS, args, 1, log_dir), name="CentralServer-Preparation")
        preparation_process.start() # 启动 Preparation 进程

        # ========== 模拟参与者节点角色 (Training Nodes) - 多进程并行 ==========
        print(f"\n>>Round {round_num} - Demo Mode: Starting Participant Nodes (Training Nodes) Processes...")
        participant_processes = []
        participant_ids = [f"participant{i+1}" for i in range(num_participants)]
        for i, participant_id in enumerate(participant_ids):
            participant_process = multiprocessing.Process(target=run_participant_node, args=(participant_clients[i], contract_abi, CONTRACT_ADDRESS, args, participant_id, log_dir), name=f"ParticipantNode-{participant_id}")
            participant_processes.append(participant_process)
            participant_process.start() # 启动参与者进程

        # ========== 每一轮的第二次运行中央服务器角色逻辑 (Demo Mode - Server Part - Aggregation) - 多进程 ==========
        print(f"\n>>Round {round_num} - Second Run: Starting Central Server (Aggregation Node) Process...")
        aggregation_process = multiprocessing.Process(target=run_central_server, args=(central_server_client, contract_abi, CONTRACT_ADDRESS, args, 2, log_dir), name="CentralServer-Aggregation")
        aggregation_process.start() # 启动 Aggregation 进程

        # ========== 每一轮的第三次运行中央服务器角色逻辑 (Demo Mode - Server Part - Evaluation) - 多进程 ==========
        print(f"\n>>Round {round_num} - Third Run: Starting Central Server (Evaluation Node) Process...")
        evaluation_process = multiprocessing.Process(target=run_central_server, args=(central_server_client, contract_abi, CONTRACT_ADDRESS, args, 3, log_dir), name="CentralServer-Evaluation")
        evaluation_process.start() # 启动 Evaluation 进程


        # ========== 等待所有进程结束 (可选) ==========
        preparation_process.join()
        print(f"\n>>Process {preparation_process.name} finished.") # 打印进程结束信息
        for participant_process in participant_processes:
            participant_process.join(timeout=300) #  !!!  设置参与者进程超时时间 (例如 5 分钟) !!!
            if participant_process.is_alive(): #  !!!  检查进程是否超时仍然存活 !!!
                print(f"\n>>Process {participant_process.name} timeout, terminating...") #  !!!  打印超时信息 !!!
                participant_process.terminate() #  !!!  强制终止超时进程 !!!
            print(f"\n>>Process {participant_process.name} finished.") # 打印进程结束信息
        aggregation_process.join()
        print(f"\n>>Process {aggregation_process.name} finished.") # 打印进程结束信息
        evaluation_process.join()
        print(f"\n>>Process {evaluation_process.name} finished.") # 打印进程结束信息
        print(f"\n>>Round {round_num} - All processes finished. Wait {wait_time} seconds before next round...") # 轮次结束信息
        print(f"\n>>[DEBUG - Main Loop - Round End] Current Round: {args.current_round}")
        time.sleep(wait_time) #  !!!  每轮结束后等待一段时间， 方便观察日志 !!!


    print("\n>>Federated Learning Demo (Multi-Node Version, Parameter Configurable, Multi-Process) Finished!") # 修改结束打印信息 -  标记为 Multi-Process