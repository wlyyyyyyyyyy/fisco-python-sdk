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
import argparse #  导入 argparse 模块
import shutil #  导入 shutil 模块，用于删除文件夹

# =====  导入 fl_utils.py 中的通用函数和类  =====
from myclient.fl_1node_utils import (
    load_mnist_data_partition,
    load_mnist_test_data,
    load_cifar10_data_partition, #  导入 CIFAR-10 数据加载函数
    load_cifar10_test_data,    #  导入 CIFAR-10 数据加载函数
    MLP,
    CNN,            #  导入 CNN 模型类
    serialize_model,
    deserialize_model,
    train_model,
    evaluate_model,
    upload_model_update,
    download_global_model,
)


# ----- 合约信息 (保持不变) -----
CONTRACT_NAME = "EvenSimplerFederatedLearning"
CONTRACT_PATH = "./contracts/EvenSimplerFederatedLearning.sol"
ABI_PATH = "./contracts/EvenSimplerFederatedLearning.abi"
BIN_PATH = "./contracts/EvenSimplerFederatedLearning.bin"
CONTRACT_NOTE_NAME = "federatedlearning_single_demo"
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


# ==========  中央服务器角色函数 (精简版 -  只负责下载模型，不再进行测试) -  修改后，第二次运行添加模型评估,  并传递日志文件夹参数 -  保持不变，添加调试信息 ==========
def run_central_server(central_server_client, contract_abi, contract_address, args, run_round, log_dir="fl_single_log"): #  添加 log_dir 参数
    if run_round == 1: # 第一次运行 (Preparation Node)
        print(f"\n>>Starting Central Server (Preparation Node)...") # 修改打印信息 -  更名为 Preparation Node
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Preparation) <<<<<<") # 修改打印信息 -  更名为 Preparation, 显示当前轮数
        print(f"\n>>Central Server (Preparation Node) in Round: {args.current_round}") # 修改打印信息 -  更名为 Preparation Node, 显示当前轮数
    elif run_round == 2: # 第二次运行 (Testing/Evaluation Node) -  添加模型评估
        print(f"\n>>Starting Central Server (Testing Node - Evaluation)...") # 修改打印信息 -  更名为 Testing Node - Evaluation
        print(f"\n>>>>> Federated Learning Round: {args.current_round} (Central Server - Evaluation) <<<<<<") # 修改打印信息 -  更名为 Evaluation, 显示当前轮数
        print(f"\n>>Central Server (Testing Node - Evaluation) in Round: {args.current_round}") # 修改打印信息 -  更名为 Testing Node - Evaluation, 显示当前轮数
    else:
        raise ValueError(f"Invalid run_round value: {run_round}. Must be 1 or 2.")


    # -----  1.  模型下载 -----
    print("\n>>Central Server: Downloading Global Model...") # 修改打印信息 -  通用描述，不再区分 Preparation/Testing Node
    downloaded_model_str = download_global_model(central_server_client, contract_abi, contract_address, None, "server") #  调用 fl_utils.py 中的 download_global_model, 并传递参数

    if downloaded_model_str:
        print("\n>>Global Model downloaded successfully.")
        model = deserialize_model(downloaded_model_str, model_type=args.model) # 使用 fl_utils.py 中的函数, 传递 model_type 参数
    else:
        print("\n>>Failed to download Global Model.")
        if run_round == 1: # 第一次运行下载失败
            print("\n>>Preparation Failed: No Global Model Downloaded.") # 修改打印信息 -  更名为 Preparation Failed
            return
        elif run_round == 2: # 第二次运行下载失败
            print("\n>>Evaluation Failed: No Global Model Downloaded for Evaluation.") # 修改打印信息 -  更名为 Evaluation Failed
            return


    if run_round == 2: # 第二次运行 -  添加模型评估
        # ----- 2. 加载测试数据集 -----
        print("\n>>Central Server (Testing Node - Evaluation): Loading Test Data...") # 修改打印信息 -  更名为 Testing Node - Evaluation
        if args.dataset == 'cifar10': #  根据参数选择数据集
            test_loader = load_cifar10_test_data()
        else: # 默认为 mnist
            test_loader = load_mnist_test_data()

        # ----- 3. 模型评估 -----
        print("\n>>Central Server (Testing Node - Evaluation): Evaluating Global Model...") # 修改打印信息 -  更名为 Testing Node - Evaluation
        print(f"\n>>[DEBUG - Central Server Evaluation Start] Round: {args.current_round}, run_round: {run_round}") # DEBUG -  输出评估开始时的轮数和 run_round
        evaluate_model(model, test_loader, log_dir=log_dir, round_num=args.current_round) # 使用 fl_utils.py 中的函数, 传递 log_dir 和 round_num
        print(f"\n>>[DEBUG - Central Server Evaluation End] Round: {args.current_round}, run_round: {run_round}") # DEBUG -  输出评估结束时的轮数和 run_round
        print(f"\n>>Central Server (Testing Node - Evaluation) Completed Round: {args.current_round}") # 修改打印信息 -  更名为 Testing Node - Evaluation, 显示当前轮数

    elif run_round == 1: # 第一次运行
        print(f"\n>>Central Server (Preparation Node) Completed Round: {args.current_round}") # 修改打印信息 -  更名为 Preparation Node, 显示当前轮数



# ==========  参与者节点角色函数 (训练和上传) - 修改后，添加轮数信息, 并传递日志文件夹参数, 修复 epochs 参数错误 - 保持不变，添加调试信息 ==========
def run_participant_node(participant_client, contract_abi, contract_address, args, log_dir="fl_single_log"): #  添加 log_dir 参数
    participant_id = "participant1" # 固定 participant_id
    # epochs = 1 #  !!!  移除固定的 epochs = 1, 使用 args.epochs
    print(f"\n>>Starting Participant Node (Training Node)...")
    print(f"\n>>>>> Federated Learning Round: {args.current_round} (Participant Node - Training) <<<<<<") # 修改打印信息 -  显示当前轮数
    print(f"\n>>Participant Node (Training Node) in Round: {args.current_round}") # 修改打印信息 -  显示当前轮数

    # ----- 1. 下载全局模型 -----
    print("\n>>Participant Node: Downloading Global Model...") # 修改打印信息 -  通用描述
    downloaded_model_str = download_global_model(participant_client, contract_abi, contract_address, None, participant_id) #  调用 fl_utils.py 中的 download_global_model, 并传递参数
    if downloaded_model_str:
        print(f"\n>>Global Model downloaded successfully.")
        model = deserialize_model(downloaded_model_str, model_type=args.model) # 使用 fl_utils.py 中的函数, 传递 model_type 参数
    else:
        print(f"\n>>Failed to download Global Model. Using local initial model.")
        if args.model == 'cnn': #  根据参数选择模型
            model = CNN()
        else: # 默认为 MLP
            model = MLP()

    # ----- 2. 本地模型训练 -----
    print("\n>>Participant Node: Loading Training Data...") # 修改打印信息 -  通用描述
    if args.dataset == 'cifar10': #  根据参数选择数据集
        train_loader = load_cifar10_data_partition()
    else: # 默认为 mnist
        train_loader = load_mnist_data_partition()
    print("\n>>Participant Node: Starting Local Model Training...") # 修改打印信息 -  通用描述
    print(f"\n>>[DEBUG - Participant Training Start] Round: {args.current_round}, Epochs: {args.epochs}") # DEBUG -  输出训练开始时的轮数和 epochs
    trained_model = train_model(model, train_loader, epochs=args.epochs, log_dir=log_dir, round_num=args.current_round) # 使用 fl_utils.py 中的函数, 传递 args.epochs, log_dir 和 round_num  !!! 使用 args.epochs
    print(f"\n>>[DEBUG - Participant Training End] Round: {args.current_round}, Epochs: {args.epochs}") # DEBUG -  输出训练结束时的轮数和 epochs
    updated_model_str = serialize_model(trained_model) # 使用 fl_utils.py 中的函数

    # ----- 3. 上传模型更新 -----
    print("\n>>Participant Node: Uploading Model Update...") # 修改打印信息 -  通用描述
    if upload_model_update(participant_client, contract_abi, contract_address, updated_model_str, participant_id): #  调用 fl_utils.py 中的 upload_model_update, 并传递参数
        print(f"\n>>Model update uploaded successfully.")
    else:
        print(f"\n>>Model update upload failed.")

    print(f"\n>>Participant Node (Training Node) Finished Round: {args.current_round}") # 修改打印信息 -  显示当前轮数



# ==========  模型聚合函数  -  保持移除状态  ==========
# ==========  federatedlearning_single.py 中不再需要 aggregate_global_model 函数的定义  ==========
# def aggregate_global_model(central_server_client, contract_abi, contract_address):
#     ...  (已删除)


# ==========  主程序入口 (简化 Demo 模式 -  单进程双客户端，多进程结构,  彻底移除 "初始测试") - 修改后，添加参数解析，并区分中央服务器的两次运行，添加多轮循环,  添加日志文件夹, 修改主循环，添加每轮评估 - 添加调试信息, 添加清空日志文件夹功能, **优化 participant_client 创建** ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Parameter Configurable Version with Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation):----------------------------------------------------------") # 修改打印信息 -  添加 "Optimized Client Creation"

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Minimal Dual Client Simulation - Single Process, Multi-Process Structure, Parameter Configurable, Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation)") # 修改打印信息，更清晰地表达当前模式


    # ==========  添加参数解析  ==========
    parser = argparse.ArgumentParser(description="Federated Learning Demo Script")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10)') #  数据集参数
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model to use (mlp or cnn)') #  模型参数
    parser.add_argument('--epochs', type=int, default=1, help='Number of local training epochs per round (default: 1)') #  epochs 参数 (虽然目前固定为 1) -  更明确的描述
    parser.add_argument('--rounds', type=int, default=2, help='Number of federated learning rounds (default: 2)') #  添加 rounds 参数，默认 2 轮
    parser.add_argument('--contract_address', type=str, default='', help='Optional: Contract address to use. If not provided, a new contract will be deployed.') #  合约地址参数
    parser.add_argument('--role', type=str, default='demo', choices=['demo', 'server', 'participant'], help='Role to run (demo, server, participant)') # 角色参数 (虽然目前固定为 demo)
    parser.add_argument('--log_dir', type=str, default='fl_single_log', help='Directory to save log files (default: fl_single_log)') #  添加 log_dir 参数，用于指定日志文件夹

    args = parser.parse_args() #  解析命令行参数

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Role: {args.role}, Contract Address: {args.contract_address}, Log Dir: {args.log_dir}") # 打印配置信息 -  添加 log_dir 参数


    CONTRACT_ADDRESS = args.contract_address #  从命令行参数获取合约地址
    role = args.role #  从命令行参数获取角色 (虽然目前 demo 模式固定为 demo)
    log_dir = args.log_dir #  从命令行参数获取日志文件夹路径


    # ========== 清空日志文件夹 (如果存在) ==========
    if os.path.exists(log_dir): #  判断日志文件夹是否存在
        print(f"\n>>Clearing existing log directory: {log_dir}") # 打印清空提示信息
        shutil.rmtree(log_dir) #  删除文件夹 (及其内容)
    else:
        print(f"\n>>Log directory not found, creating: {log_dir}") # 打印创建提示信息

    # ========== 创建日志文件夹 ==========
    os.makedirs(log_dir, exist_ok=True) #  确保日志文件夹存在


    central_server_client = Bcos3Client() #  创建中央服务器客户端实例
    participant_client = Bcos3Client() #  !!!  在循环外创建 participant_client  !!!


    # ========== 部署合约 (Demo Mode - Server Part) - 部署合约只在第一轮之前进行 ==========
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
        print(f"\n>>[DEBUG - Central Server Preparation Start] Round: {args.current_round}, run_round: 1") # DEBUG -  输出 Preparation 开始时的轮数和 run_round
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_round=1, log_dir=log_dir) #  顺序执行中央服务器逻辑 (Preparation), 传递 args, run_round=1,  传递 log_dir
        print(f"\n>>[DEBUG - Central Server Preparation End] Round: {args.current_round}, run_round: 1") # DEBUG -  输出 Preparation 结束时的轮数和 run_round
        print(f"\n>>Round {round_num} - First Run: Demo Mode - Central Server (Preparation Node) process finished.") # 修改打印信息 -  更名为 Preparation Node


        # ========== 模拟参与者节点角色 (Demo Mode - Participant Part - Training and Upload) ==========
        # participant_client = Bcos3Client() #  !!!  移除循环内的 participant_client 创建  !!!
        print(f"\n>>Round {round_num} - Demo Mode: Participant Node (Training Node) process starting...") # 修改打印信息
        print(f"\n>>[DEBUG - Participant Node Start] Round: {args.current_round}") # DEBUG -  输出 Participant Node 开始时的轮数
        run_participant_node(participant_client, contract_abi, CONTRACT_ADDRESS, args, log_dir=log_dir) #  顺序执行参与者节点逻辑 (Training and Upload), 传递 args,  传递 log_dir
        print(f"\n>>[DEBUG - Participant Node End] Round: {args.current_round}") # DEBUG -  输出 Participant Node 结束时的轮数
        print(f"\n>>Round {round_num} - Demo Mode: Participant Node (Training Node) process finished.") # 修改打印信息


        # ==========  每一轮的第二次运行中央服务器角色逻辑 (Demo Mode - Server Part - Evaluation) -  每轮训练后都评估 ==========
        print(f"\n>>Round {round_num} - Second Run: Central Server (Testing Node - Evaluation) - Model Evaluation after Round {round_num} Training...") # 修改打印信息 -  更准确的描述，添加 "Model Evaluation after Round {round_num} Training"
        print(f"\n>>[DEBUG - Central Server Evaluation Start] Round: {args.current_round}, run_round: 2") # DEBUG -  输出 Evaluation 开始时的轮数和 run_round
        run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_round=2, log_dir=log_dir) #  每轮训练后都进行模型评估, 传递 args, run_round=2, 传递 log_dir
        print(f"\n>>[DEBUG - Central Server Evaluation End] Round: {args.current_round}, run_round: 2") # DEBUG -  输出 Evaluation 结束时的轮数和 run_round
        print(f"\n>>Round {round_num} - Second Run: Demo Mode - Central Server (Testing Node - Evaluation) process finished.") # 修改打印信息 -  更名为 Testing Node - Evaluation
        print(f"\n>>[DEBUG - Main Loop - Round End] Current Round: {args.current_round}") # DEBUG -  输出每轮循环结束时的轮数


    # ==========  最终评估 (代码重复，但为了逻辑清晰，保留) -  所有轮次结束后，实际上在循环的最后一轮已经评估过，这里可以移除，或者保留并修改打印信息 ==========
    # print(f"\n{'='*20} Federated Learning Final Evaluation {'='*20}") #  添加最终评估分隔符
    # print("\n>>Final Evaluation Run: Central Server (Testing Node - Evaluation) - Model Evaluation after Training...") # 修改打印信息 -  更准确的描述，添加 "Final Evaluation"
    # args.current_round = args.rounds + 1 #  评估轮次设置为 rounds + 1，用于区分评估轮次  !!!  不再需要，因为评估已经在循环中进行
    # run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS, args, run_round=2, log_dir=log_dir) #  最后进行模型评估, 传递 args, run_round=2, 传递 log_dir  !!!  不再需要，评估已经在循环中进行
    # print("\n>>Final Evaluation Run: Demo Mode - Central Server (Testing Node - Evaluation) process finished.") # 修改打印信息 -  更名为 Testing Node - Evaluation


    print("\n>>Federated Learning Demo (Parameter Configurable Version with Multi-Round Training, Evaluation and Logging, Per-Round Evaluation, Clear Log Folder, Optimized Client Creation) Finished!") # 修改打印信息，更清晰地表达当前模式