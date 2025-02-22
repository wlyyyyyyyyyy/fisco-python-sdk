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
import torch #  导入 torch 模块  <---  Add this line

# =====  导入 fl_2nodes_utils.py 中的通用函数和类  =====
from myclient.fl_2nodes_utils import (
    log_operation,
    load_mnist_data_partition_2nodes,
    load_mnist_test_data,
    load_cifar10_data_partition_2nodes,
    load_cifar10_test_data,
    MLP,
    CNN,
    serialize_model,
    deserialize_model,
    train_model,
    evaluate_model,
    upload_model_update,
    download_global_model,
)

# ----- 合约信息 (保持不变，但合约名需要更新为 EnhancedFederatedLearning) -----
CONTRACT_NAME = "EnhancedFederatedLearning" #  !!!  更新合约名称 !!!
CONTRACT_PATH = "./contracts/EnhancedFederatedLearning.sol" #  !!!  更新合约路径 !!!
ABI_PATH = "./contracts/EnhancedFederatedLearning.abi" #  !!!  更新 ABI 路径 !!!
BIN_PATH = "./contracts/EnhancedFederatedLearning.bin" #  !!!  更新 BIN 路径 !!!
CONTRACT_NOTE_NAME = "federatedlearning_2nodes_demo" #  !!!  更新 Contract Note 名称 !!!
CONTRACT_ADDRESS = "" #  合约地址可以从命令行参数传入，所以这里初始化为空字符串

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


# ==========  中央服务器角色函数 (修改后，实现模型聚合和更新) ==========
def run_central_server(central_server_client, contract_abi, contract_address, args, run_mode, log_dir="fl_2nodes_log"): #  参数名修改为 run_mode
    if run_mode == 1: # 第一次运行 (Preparation Node) -  run_mode == 1 代表 Preparation 模式
        print(f"\n>>Starting Central Server (Preparation Node)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Preparation) <<<<<<")
        print(f"\n>>Central Server (Preparation Node) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "preparation_start", "Central Server Preparation Node started.")
    elif run_mode == 2: # 第二次运行 (Aggregation Node) - run_mode == 2 代表 Aggregation 模式
        print(f"\n>>Starting Central Server (Aggregation Node)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Aggregation) <<<<<<")
        print(f"\n>>Central Server (Aggregation Node) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "aggregation_start", "Central Server Aggregation Node started.")
    elif run_mode == 3: # 第三次运行 (Testing/Evaluation Node) - run_mode == 3 代表 Evaluation 模式
        print(f"\n>>Starting Central Server (Testing Node - Evaluation)...")
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Evaluation) <<<<<<")
        print(f"\n>>Central Server (Testing Node - Evaluation) in Round: {args.current_round}")
        log_operation(log_dir, args.current_round, "server", "evaluation_start", "Central Server Evaluation Node started.")
    else:
        raise ValueError(f"Invalid run_mode value: {run_mode}. Must be 1, 2 or 3.") #  错误提示信息中的参数名也改为 run_mode


    # -----  1. 模型下载 (Preparation Node 移除下载, Aggregation, Evaluation Node 都需要下载全局模型) -----
    if run_mode in [2, 3]: # Preparation Node 移除下载,  Aggregation, Evaluation Node 才下载模型  !!! 修改条件为 run_mode !!!
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
            if run_mode == 2: # 第二次运行下载失败 -  !!! 修改条件为 run_mode !!!
                print("\n>>Aggregation Failed: No Global Model Downloaded for Aggregation.")
                log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to global model download failure.")
                return
            elif run_mode == 3: # 第三次运行下载失败 -  !!! 修改条件为 run_mode !!!
                print("\n>>Evaluation Failed: No Global Model Downloaded for Evaluation.")
                log_operation(log_dir, args.current_round, "server", "evaluation_fail", "Evaluation failed due to global model download failure.")
                return


    if run_mode == 2: # 第二次运行 (Aggregation Node) -  模型聚合和更新 -  !!! 修改条件为 run_mode !!!
        # ----- 2. 下载参与者模型更新 -----
        print("\n>>Central Server (Aggregation Node): Downloading Participant Model Updates...")
        log_operation(log_dir, args.current_round, "server", "download_participant_updates_start", "Central Server starts downloading participant model updates.")
        participant_updates_json_str = central_server_client.call(contract_address, contract_abi, "getParticipantUpdates", [args.current_round])[0]
        if participant_updates_json_str:
            print("\n>>Participant Model Updates downloaded successfully (JSON String).")
            log_operation(log_dir, args.current_round, "server", "download_participant_updates_success", "Participant model updates downloaded successfully.")
            participant_updates = json.loads(participant_updates_json_str)
            print(f"\n>>[DEBUG - Aggregation Node] Participant Updates (JSON): \n{json.dumps(participant_updates, indent=2)}")
        else:
            print("\n>>Failed to download Participant Model Updates.")
            log_operation(log_dir, args.current_round, "server", "download_participant_updates_fail", "Failed to download participant model updates.")
            print("\n>>Aggregation Failed: No Participant Model Updates Downloaded.")
            log_operation(log_dir, args.current_round, "server", "aggregation_fail", "Aggregation failed due to participant model updates download failure.")
            return

        # ----- 3. 模型聚合 (简单的平均聚合) -----
        print("\n>>Central Server (Aggregation Node): Aggregating Model Updates...")
        log_operation(log_dir, args.current_round, "server", "aggregation_model_start", "Central Server starts aggregating model updates.")
        aggregated_model = aggregate_global_model(global_model, participant_updates, model_type=args.model)
        aggregated_model_str = serialize_model(aggregated_model)
        log_operation(log_dir, args.current_round, "server", "aggregation_model_success", "Global model aggregation finished.")

        # ----- 4. 上传聚合后的全局模型 -----
        print("\n>>Central Server (Aggregation Node): Uploading Aggregated Global Model...")
        log_operation(log_dir, args.current_round, "server", "upload_aggregated_model_start", "Central Server starts uploading aggregated global model.")
        if upload_global_model(central_server_client, contract_abi, contract_address, aggregated_model_str):
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


    elif run_mode == 3: # 第三次运行 (Testing Node - Evaluation) -  模型评估节点 -  !!! 修改条件为 run_mode !!!
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


    elif run_mode == 1: # 第一次运行 (Preparation Node) -  !!! 修改条件为 run_mode !!!
        log_operation(log_dir, args.current_round, "server", "preparation_complete", f"Central Server Preparation Node Completed Round: {args.current_round}")
        print(f"\n>>Central Server (Preparation Node) Completed Round: {args.current_round}")


# ==========  参与者节点角色函数 (训练和上传) - 修改后，使用 2 节点数据加载函数，传递 participant_id ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, participant_id="participant1", log_dir="fl_2nodes_log"):
    # ... (函数开始部分，打印信息) ...
    log_operation(log_dir, args.current_round, participant_id, "training_node_start", f"Participant Node {participant_id} Training Node started.") # 添加操作日志

    # ----- 1. 下载全局模型 -----
    print(f"\n>>Participant Node ({participant_id}): Downloading Global Model...") #  打印 participant_id
    log_operation(log_dir, args.current_round, participant_id, "download_global_model_start", f"Participant Node {participant_id} starts downloading global model.") #  添加操作日志
    downloaded_model_str = download_global_model(participant_client, contract_abi, contract_address, None, participant_id) #  角色名使用 participant_id
    if downloaded_model_str:
        print(f"\n>>Global Model downloaded successfully (for Participant: {participant_id}).") #  打印 participant_id
        log_operation(log_dir, args.current_round, participant_id, "download_global_model_success", f"Participant Node {participant_id} global model downloaded successfully.") #  添加操作日志
        model = deserialize_model(downloaded_model_str, model_type=args.model)
    else:
        print(f"\n>>Failed to download Global Model (for Participant: {participant_id}). Using local initial model.") #  打印 participant_id
        log_operation(log_dir, args.current_round, participant_id, "download_global_model_fail", f"Participant Node {participant_id} failed to download global model, using local initial model.") #  添加操作日志
        if args.model == 'cnn':
            model = CNN()
        else:
            model = MLP()

    # ----- 2. 本地模型训练 -----
    print(f"\n>>Participant Node ({participant_id}): Loading Training Data...") #  打印 participant_id
    if args.dataset == 'cifar10': #  根据参数选择数据集
        train_loader = load_cifar10_data_partition_2nodes(participant_id=participant_id) #  使用 CIFAR-10 的 2 节点数据加载函数, 传递 participant_id
    else: # 默认为 mnist
        train_loader = load_mnist_data_partition_2nodes(participant_id=participant_id) #  使用 MNIST 的 2 节点数据加载函数, 传递 participant_id
    print(f"\n>>Participant Node ({participant_id}): Starting Local Model Training...") #  打印 participant_id
    log_operation(log_dir, args.current_round, participant_id, "local_training_start", f"Participant Node {participant_id} local model training started.") #  添加操作日志
    trained_model = train_model(model, train_loader, epochs=args.epochs, log_dir=log_dir, round_num=args.current_round, participant_id=participant_id) #  传递 participant_id
    updated_model_str = serialize_model(trained_model)
    log_operation(log_dir, args.current_round, participant_id, "local_training_success", f"Participant Node {participant_id} local model training finished.") #  添加操作日志

    # ----- 3. 上传模型更新 -----
    print(f"\n>>Participant Node ({participant_id}): Uploading Model Update...") #  打印 participant_id
    log_operation(log_dir, args.current_round, participant_id, "upload_model_update_start", f"Participant Node {participant_id} starts uploading model update.") #  添加操作日志
    if upload_model_update(participant_client, contract_abi, contract_address, updated_model_str, participant_id, args.current_round): #  角色名使用 participant_id, 传递 round_num
        print(f"\n>>Model update uploaded successfully (from Participant: {participant_id}).") #  打印 participant_id
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_success", f"Participant Node {participant_id} model update uploaded successfully.") #  添加操作日志
    else:
        print(f"\n>>Model update upload failed (from Participant: {participant_id}).") #  打印 participant_id
        log_operation(log_dir, args.current_round, participant_id, "upload_model_update_fail", f"Participant Node {participant_id} model update upload failed.") #  添加操作日志

    log_operation(log_dir, args.current_round, participant_id, "training_node_complete", f"Participant Node {participant_id} Training Node Finished Round: {args.current_round}") # 添加操作日志
    print(f"\n>>Participant Node ({participant_id} - Training Node) Finished Round: {args.current_round}") #  打印 participant_id 和轮数



# ==========  模型聚合函数 (简单的平均聚合) -  修改后，适配 participant_updates JSON 格式，并处理模型类型参数 ==========
def aggregate_global_model(global_model, participant_updates, model_type='cnn'): #  添加 model_type 参数，默认为 'cnn'
    print("\n>>Starting Global Model Aggregation...")
    aggregated_state_dict = {}
    participant_models = []

    for update in participant_updates: # 遍历 participant_updates JSON 数组
        participant_id = update['participantId'] # 从 JSON 中获取 participantId
        model_update_str = update['modelUpdate'] # 从 JSON 中获取 modelUpdate (模型字符串)
        print(f"\n>>Deserializing model update from participant: {participant_id}...") # 打印 participantId
        participant_model = deserialize_model(model_update_str, model_type=model_type) #  反序列化参与者模型, 传递 model_type 参数
        participant_models.append(participant_model) # 添加到参与者模型列表


    # -----  简单的平均聚合  -----
    print("\n>>Performing Averaging Aggregation...")
    with torch.no_grad():
        # 初始化聚合模型的状态字典
        for name, param in global_model.state_dict().items():
            aggregated_state_dict[name] = torch.zeros_like(param) #  使用全局模型的形状初始化为零张量

        # 累加所有参与者模型的权重
        for participant_model in participant_models:
            for name, param in participant_model.state_dict().items():
                aggregated_state_dict[name] += param

        # 平均权重
        num_participants = len(participant_models)
        for name, param in aggregated_state_dict.items():
            aggregated_state_dict[name] /= num_participants

        # 将聚合后的状态字典加载到全局模型
        global_model.load_state_dict(aggregated_state_dict) #  加载聚合后的权重

    print("\n>>Global Model Aggregation Finished!")
    return global_model


# ==========  全局模型上传函数 (新增) ==========
def upload_global_model(central_server_client, contract_abi, contract_address, global_model_str):
    print(f"\n>>Uploading Aggregated Global Model (from Role: server)...") #  角色名固定为 "server"
    to_address = contract_address
    fn_name = "updateModel"
    args = [global_model_str, 0, "server"] #  轮数设置为 0, 角色名设置为 "server"
    receipt = central_server_client.sendRawTransaction(to_address, contract_abi, fn_name, args) #  调用 updateModel 函数
    if receipt is not None and receipt["status"] == 0:
        print(f"Upload Aggregated Global Model Success (from Role: server), receipt: {receipt}") #  角色名固定为 "server"
        return True
    else:
        print(f"Upload Aggregated Global Model Failed (from Role: server), receipt: {receipt}") #  角色名固定为 "server"
        return False



# ==========  主程序入口 (Demo 模式 -  单进程双客户端，多进程结构,  彻底移除 "初始测试") - 修改后，添加参数解析，并区分中央服务器的三次运行，添加多轮循环,  添加日志文件夹, 修改主循环，添加每轮评估 - 添加调试信息, 添加清空日志文件夹功能, 优化 participant_client 创建,  适配双节点流程  ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (2-Node Version, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server):----------------------------------------------------------") # 修改打印信息 -  标记为 2 节点版本和 3-Run 中央服务器

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Minimal Dual Client Simulation - Single Process, Multi-Process Structure, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server)") # 修改打印信息，更清晰地表达当前模式


    # ==========  添加参数解析 (复用 federatedlearning_single.py 的参数) ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script (2-Node Version)") # 修改 description
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use (mlp or cnn)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round (default: 1)')
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds (default: 2)') #  轮数参数保持默认 2 轮
    parser.add_argument('--contract_address', type=str, default='', help='Optional: Contract address to use. If not provided, a new contract will be deployed.')
    parser.add_argument('--role', type=str, default='demo', choices=['demo', 'server', 'participant'], help='Role to run (demo, server, participant)') # 角色参数 (虽然 demo 模式固定为 demo)
    parser.add_argument('--log_dir', type=str, default='fl_2nodes_log', help='Directory to save log files (default: fl_2nodes_log)') #  日志文件夹参数修改为 fl_2nodes_log

    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Role: {args.role}, Contract Address: {args.contract_address}, Log Dir: {args.log_dir}") # 打印配置信息 -  日志文件夹参数修改为 fl_2nodes_log


    CONTRACT_ADDRESS = args.contract_address
    role = args.role #  从命令行参数获取角色 (虽然 demo 模式固定为 demo)
    log_dir = args.log_dir #  从命令行参数获取日志文件夹路径


    # ========== 清空日志文件夹 (如果存在) ==========
    if os.path.exists(log_dir):
        print(f"\n>>Clearing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    else:
        print(f"\n>>Log directory not found, creating: {log_dir}")

    # ========== 创建日志文件夹 ==========
    os.makedirs(log_dir, exist_ok=True)


    central_server_client = Bcos3Client() #  创建中央服务器客户端实例
    participant_client_1 = Bcos3Client() #  创建参与者客户端实例 1
    participant_client_2 = Bcos3Client() #  创建参与者客户端实例 2  !!!  创建第二个参与者客户端


    # ========== 部署合约 (Demo Mode - Server Part - Preparation) - 部署合约只在第一轮之前进行 ==========
    if not CONTRACT_ADDRESS:
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Deploying contract...") # 修改打印信息 -  更名为 Preparation Node
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = central_server_client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"]
        print(f"Demo Mode - Central Server (Preparation Node): Deploy contract success, contract address: {CONTRACT_ADDRESS}") # 修改打印信息 -  更名为 Preparation Node
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
    else:
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Using existing contract at: {CONTRACT_ADDRESS}") # 修改打印信息 -  更名为 Preparation Node


    # ==========  多轮联邦学习循环  ==========
    print(f"\n>>[DEBUG - Main Loop Start] Rounds: {args.rounds}") # DEBUG -  输出总轮数
    for round_num in range(1, args.rounds + 1): #  从第 1 轮循环到 rounds 轮
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}") #  添加轮次分隔符
        args.current_round = round_num #  将当前轮数添加到 args 中，方便传递给函数
        print(f"\n>>[DEBUG - Main Loop - Round Start] Current Round: {args.current_round}") # DEBUG -  输出每轮循环开始时的轮数

        # ========== 每一轮的第一次运行中央服务器角色逻辑 (Demo Mode - Server Part - Preparation) ==========
        print(f"\n>>Round {round_num} - First Run: Central Server (Preparation Node) - Model Preparation...") # 修改打印信息 -  更准确的描述
        print(f"\n>>[DEBUG - Central Server Preparation Start] Round: {args.current_round}, run_mode: 1") #  修改为 run_mode
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=1, log_dir=log_dir) #  顺序执行中央服务器逻辑 (Preparation), 传递 args, run_mode=1,  传递 log_dir
        print(f"\n>>[DEBUG - Central Server Preparation End] Round: {args.current_round}, run_mode: 1") # DEBUG -  输出 Preparation 结束时的轮数和 run_mode
        print(f"\n>>Round {round_num} - First Run: Demo Mode - Central Server (Preparation Node) process finished.") # 修改打印信息 -  更名为 Preparation Node


        # ========== 模拟参与者节点角色 (Demo Mode - Participant Part - Training and Upload) -  模拟两个参与者节点并行训练 ==========
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training Nodes) process starting...") # 修改打印信息 -  复数 Nodes
        print(f"\n>>[DEBUG - Participant Nodes Start] Round: {args.current_round}") # DEBUG -  输出 Participant Nodes 开始时的轮数

        # -----  模拟两个参与者节点并行训练 -----
        participant_id_1 = "participant1" #  定义参与者 1 ID
        participant_id_2 = "participant2" #  定义参与者 2 ID
        print(f"\n>>Starting Participant Node ({participant_id_1})...") #  打印参与者 1 启动信息
        run_participant_node(participant_client_1, contract_abi, CONTRACT_ADDRESS, args, participant_id=participant_id_1, log_dir=log_dir) #  运行参与者节点 1, 传递 participant_id_1

        print(f"\n>>Starting Participant Node ({participant_id_2})...") #  打印参与者 2 启动信息
        run_participant_node(participant_client_2, contract_abi, CONTRACT_ADDRESS, args, participant_id=participant_id_2, log_dir=log_dir) #  运行参与者节点 2, 传递 participant_id_2

        print(f"\n>>[DEBUG - Participant Nodes End] Round: {args.current_round}") # DEBUG -  输出 Participant Nodes 结束时的轮数
        print(f"\n>>Round {round_num} - Demo Mode: Participant Nodes (Training Nodes) process finished.") # 修改打印信息 -  复数 Nodes


        # ==========  每一轮的第二次运行中央服务器角色逻辑 (Demo Mode - Server Part - Aggregation) ==========
        print(f"\n>>Round {round_num} - Second Run: Central Server (Aggregation Node) - Model Aggregation...") # 修改打印信息 -  更准确的描述
        print(f"\n>>[DEBUG - Central Server Aggregation Start] Round: {args.current_round}, run_mode: 2") # 修改为 run_mode
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=2, log_dir=log_dir) #  顺序执行中央服务器逻辑 (Aggregation), 传递 args, run_mode=2, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Aggregation End] Round: {args.current_round}, run_mode: 2") # DEBUG -  输出 Aggregation 结束时的轮数和 run_mode
        print(f"\n>>Round {round_num} - Second Run: Demo Mode - Central Server (Aggregation Node) process finished.") # 修改打印信息 -  更名为 Aggregation Node


        # ==========  每一轮的第三次运行中央服务器角色逻辑 (Demo Mode - Server Part - Evaluation) -  每轮训练后都评估 ==========
        print(f"\n>>Round {round_num} - Third Run: Central Server (Testing Node - Evaluation) - Model Evaluation after Round {round_num} Aggregation...") # 修改打印信息 -  更准确的描述,  添加 "Model Evaluation after Round {round_num} Aggregation"
        print(f"\n>>[DEBUG - Central Server Evaluation Start] Round: {args.current_round}, run_mode: 3") # 修改为 run_mode
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_mode=3, log_dir=log_dir) #  每轮训练后都进行模型评估, 传递 args, run_mode=3, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Evaluation End] Round: {args.current_round}, run_mode: 3") # DEBUG -  输出 Evaluation 结束时的轮数和 run_mode
        print(f"\n>>Round {round_num} - Third Run: Demo Mode - Central Server (Testing Node - Evaluation) process finished.") # 修改打印信息 -  更名为 Testing Node - Evaluation
        print(f"\n>>[DEBUG - Main Loop - Round End] Current Round: {args.current_round}") # DEBUG -  输出每轮循环结束时的轮数


    print("\n>>Federated Learning Demo (2-Node Version, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation, 3-Run Central Server) Finished!") # 修改打印信息，更清晰地表达当前模式
