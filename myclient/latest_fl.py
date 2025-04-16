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
import copy
from client_config import client_config as ClientConfig
import time
import random


# ===== 导入我们修改后的 LATEST utils =====
from latest_utils import (
    # ... (utils imports remain the same) ...
    log_operation,
    load_mnist_data_partition_multi, load_mnist_test_data,
    load_cifar10_data_partition_multi, load_cifar10_test_data,
    MLP, CNN,
    serialize_model, deserialize_model,
    train_model,
    aggregate_global_model,
    calculate_keccak256_onchain,
    submit_update_to_contract, get_updates_from_contract,
    submit_hash_to_contract, get_hash_from_contract,
    get_latest_round_from_contract, get_participant_ids_from_contract,
    evaluate_model
)

# ----- 合约信息 -----
CONTRACT_NAME = "Latest"
CONTRACT_PATH = "./contracts/Latest.sol"
ABI_PATH = f"./contracts/{CONTRACT_NAME}.abi"
BIN_PATH = f"./contracts/{CONTRACT_NAME}.bin"
CONTRACT_NOTE_NAME = "federatedlearning_latest_onchain_v10" # New name
# CONTRACT_ADDRESS = ""

# ----- Demo 模式配置 -----
WAIT_TIME_SECONDS = 1
MAX_WAIT_ATTEMPTS = 12

# ----- 自动编译合约 -----
if not (os.path.exists(ABI_PATH) and os.path.exists(BIN_PATH)):
    print(f"ABI or BIN file not found, compiling contract: {CONTRACT_PATH}")
    Compiler.compile_file(CONTRACT_PATH)

abi_file = ABI_PATH
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

# ----- 主程序入口 -----
if __name__ == "__main__":

    print("\n>>Starting Decentralized FL Demo (Original Client Class Handling):----------------------------------------------------------")

    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Decentralized FL Demo (Original Client Class)")
    # ... (保持不变) ...
    parser.add_argument('--dataset',type=str,default='mnist',choices=['mnist','cifar10'])
    parser.add_argument('--model',type=str,default='mlp',choices=['mlp','cnn'])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--log_dir',type=str,default='fl_latest') # New log
    parser.add_argument('--num_participants',type=int,default=3)
    parser.add_argument('--address',type=str,default='')
    args = parser.parse_args()

    print(f"\n>>Running with configurations: Dataset: {args.dataset}, Model: {args.model}, Epochs: {args.epochs}, Rounds: {args.rounds}, Log Dir: {args.log_dir}, Num Participants: {args.num_participants}")

    log_dir = args.log_dir
    num_participants = args.num_participants
    participant_ids = [f"participant{i+1}" for i in range(num_participants)]

    # ========== 日志文件夹处理 ==========
    if os.path.exists(log_dir):
        print(f"\n>>Clearing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    else:
        print(f"\n>>Log directory not found, creating: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    # ========== 设置 BCOS 客户端 (恢复原始类实例化) ==========
    print("\n>> Setting up BCOS Clients...")
    try:
        deploy_client = Bcos3Client() #  创建中央服务器客户端实例
        participant_clients = [] #  !!!  使用列表存储多个参与者客户端实例 !!!
        for i in range(num_participants): #  !!!  根据 num_participants 创建多个参与者客户端 !!!
            client_config=ClientConfig()
            client_config.account_keyfile = f"client{i}.keystore" #!!!  根据参与者 ID 生成 keystore 文件名!!!
            client_config.account_password = f"{i}"*6 #!!!  固定密码!!!
            participant_clients.append(Bcos3Client(client_config)) #  为每个参与者创建一个客户端实例

    except NameError: # 如果 ClientConfig 类未定义
        print("[ERROR] 'ClientConfig' class not found. Make sure it's defined in/imported from 'client_config.py'.")
        sys.exit(1)
    except TypeError as e: # 如果 ClientConfig() 调用失败 (例如需要参数)
        print(f"[ERROR] Failed to instantiate ClientConfig: {e}. Check its __init__ method.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Client setup failed: {e}"); traceback.print_exc(); sys.exit(1)


    # ========== 部署合约 (Demo Mode - Server Part - Preparation) - 部署合约只在第一轮之前进行 ==========
    CONTRACT_ADDRESS = args.address # 从命令行参数获取合约地址
    if not CONTRACT_ADDRESS: # 使用硬编码的 CONTRACT_ADDRESS，这里判断可以移除，直接部署
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = deploy_client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"] # 更新 CONTRACT_ADDRESS 为实际部署地址
    else: # 即使有硬编码地址，也提示正在使用
        print(f"\n>>Demo Mode - Central Server (Preparation Node): Using existing contract at hardcoded address: {CONTRACT_ADDRESS}")

    # --- START OF FILE myclient/latest_fl.py --- (Corrected Data Loading Call)
# --- (前面的代码保持不变，从这里继续) ---

    # ========== 初始化模型 ==========
    print("\n>> Initializing models..."); participant_models = {};
    try:
        initial_model_object = None;
        if args.model=='mlp': initial_model_object = MLP()
        elif args.model=='cnn': initial_model_object = CNN(num_classes=10)
        else: raise ValueError(f"Bad model: {args.model}")
        for p_id in participant_ids: participant_models[p_id] = copy.deepcopy(initial_model_object) # 模型对象仍需拷贝
        print(">> Initial models created.")
    except Exception as e: print(f"[ERROR] Model init failed: {e}"); sys.exit(1)

    # ========== 加载测试数据 ==========
    test_loader = None; print("\n>> Loading test data...");
    if args.dataset=='cifar10': test_loader=load_cifar10_test_data()
    else: test_loader=load_mnist_test_data()
    if test_loader is None: print("[WARN] Test data load failed.")

    # ========== 多轮联邦学习循环 ==========
    print("\n" + "="*20 + " Starting Rounds " + "="*20)
    round_estimated_durations = {} # 用于存储每轮估计的并行耗时
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*20} Federated Learning Round: {round_num} {'='*20}"); t_start=time.time(); agg_model_obj=None; leader_hash=None;

        # --- 1. 本地训练 & 提交更新 ---
        print(f"\n>> Round {round_num}: Participants Training...")
        participant_training_times = {}
        for i, p_id in enumerate(participant_ids):
            print(f"\n>> Starting Participant {p_id}...")
            client = participant_clients[i] # 从列表获取客户端
            model = participant_models[p_id]
            if model is None: continue # 基本检查

            print(f">> P:{p_id} Loading Data...")
            if args.dataset=='cifar10': loader=load_cifar10_data_partition_multi(participant_id=p_id,total_participants=num_participants)
            else: loader=load_mnist_data_partition_multi(participant_id=p_id,total_participants=num_participants)

            print(f">> P:{p_id} Starting Training...")
            train_start_time = time.time()
            trained_model = train_model(model, loader, args.epochs, log_dir, round_num, p_id)
            train_end_time = time.time()
            if trained_model is None: continue
            participant_training_times[p_id] = train_end_time - train_start_time

            print(f">> P:{p_id} Uploading Update...")
            upload_start_time = time.time()
            update_str = serialize_model(trained_model)
            success, _ = submit_update_to_contract(client,contract_abi,CONTRACT_ADDRESS,update_str,round_num,p_id)
            upload_end_time = time.time()
            print(f">> P:{p_id} Upload Success: {success}")
            participant_training_times[p_id] += (upload_end_time - upload_start_time) # Include upload time in the participant's time

        print(f"\n>> Round {round_num}: Participants Training Finished.")

        # --- 2. 等待 & 本地聚合 & 提交哈希 ---
        
        print(f"\n>> Round {round_num}: Aggregation Phase...")
        aggregator_id = participant_ids[0]; aggregator_client = participant_clients[0];
        print(f">> {aggregator_id} leads aggregation...");

        # a. Wait for updates
        print(f">> Waiting for {num_participants} updates..."); update_dict=None;
        for attempt in range(MAX_WAIT_ATTEMPTS):
            ids = get_participant_ids_from_contract(aggregator_client,contract_abi,CONTRACT_ADDRESS,round_num)
            print(f"   Att:{attempt+1} Found:{len(ids)}/{num_participants}")
            if len(ids) >= num_participants:
                update_dict = get_updates_from_contract(aggregator_client,contract_abi,CONTRACT_ADDRESS,round_num)
                if update_dict and len(update_dict) >= num_participants: break
            time.sleep(WAIT_TIME_SECONDS)
        else: print(f"[ERROR] Timeout updates R:{round_num}"); print("!! Skip round !!"); continue

        # b. Local Aggregation
        aggregation_start_time = time.time()
        print(">> Aggregating models locally...")
        agg_model_obj = aggregate_global_model(update_dict,args.model,args.dataset)
        if agg_model_obj is None: print("[ERROR] Agg failed R:{round_num}"); print("!! Skip round !!"); continue
        print(">> Aggregation complete.")

        # c. Calculate Hash (On-Chain) & Submit
        agg_model_str = serialize_model(agg_model_obj)
        print(">> Calculating hash via contract...")
        hash_start_time = time.time()
        leader_hash = calculate_keccak256_onchain(aggregator_client,contract_abi,CONTRACT_ADDRESS,agg_model_str)
        hash_end_time = time.time()
        if leader_hash is None: print("[ERROR] Hash calc failed R:{round_num}"); print("!! Skip round !!"); continue
        print(f">> Hash calculated: {leader_hash.hex()}")
        print(">> Submitting hash...")
        submit_hash_start_time = time.time()
        success, _ = submit_hash_to_contract(aggregator_client,contract_abi,CONTRACT_ADDRESS,round_num,leader_hash,aggregator_id)
        submit_hash_end_time = time.time()
        print(f">> Hash submission success: {success}")
        aggregation_end_time = time.time()

        # --- 3. 验证阶段 ---
        
        print(f"\n>> Round {round_num}: Verification Phase...")
        all_ok=True; print(">> Wait before verify..."); time.sleep(1);
        for i, p_id in enumerate(participant_ids):
            verification_start_time = time.time()
            client = participant_clients[i] # 从列表获取客户端
            print(f"-- P:{p_id} verifying...")
            chain_hash = get_hash_from_contract(client,contract_abi,CONTRACT_ADDRESS,round_num)
            if chain_hash is None: print(f"[WARN] {p_id} get hash fail"); all_ok=False; continue
            if leader_hash is None: print(f"[WARN] {p_id} skip verify, leader hash bad"); all_ok=False; continue
            if chain_hash == leader_hash: print(f">> {p_id}: Verify OK!")
            else: print(f"!! {p_id}: VERIFY FAIL R:{round_num} !! MISMATCH"); all_ok=False
            verification_end_time = time.time()

        # --- 4. 更新本地模型 ---
        print(f"\n>> Round {round_num}: Update local models...")
        if agg_model_obj:
            # *** 模型对象仍然需要深拷贝 ***
            for p_id in participant_ids: participant_models[p_id] = copy.deepcopy(agg_model_obj)
            print(">> Local models updated.")
            if test_loader:
                print(f">> Round {round_num}: Evaluating model...")
                evaluate_model(agg_model_obj,test_loader,log_dir,round_num,f"agg_r{round_num}")
        else: print("[WARN] Agg failed, models not updated.")

        # --- Round End ---
        round_end_time_full = time.time()
        round_duration_full = round_end_time_full - t_start
        print(f"\n{'='*10} R:{round_num} Finished. Full Dur:{round_duration_full:.2f}s {'='*10}")
      
        # --- Estimate Parallel Time ---
        max_participant_time = max(participant_training_times.values()) if participant_training_times else 0
        sequential_time = (aggregation_end_time - aggregation_start_time) + (verification_end_time - verification_start_time)
        estimated_parallel_time = max_participant_time + sequential_time
        round_estimated_durations[round_num] = estimated_parallel_time
        print(f">> Round {round_num}: Estimated Parallel Dur:{estimated_parallel_time:.2f}s")


        # --- 5. 等待确认 ---
        print(f"\n--- R:{round_num}: Wait confirm ---"); confirmed=False
        for _ in range(MAX_WAIT_ATTEMPTS // 2):
             latest_round = get_latest_round_from_contract(deploy_client,contract_abi,CONTRACT_ADDRESS)
             if latest_round >= round_num: print(f">> R:{round_num} confirmed (Latest:{latest_round})."); confirmed=True; break
             else: print(f"   Wait confirm R:{round_num}. Latest:{latest_round}. Wait..."); time.sleep(WAIT_TIME_SECONDS)
        if not confirmed: print(f"[WARN] Timeout confirm R:{round_num}")

        # print(f"\n>>[DEBUG - Main Loop - Round End] Current Round: {round_num}") # Optional Debug


    # ========== 结束 ==========
    print("\n>> Federated Learning Demo (Original Client Handling) Finished!")
    final_round = get_latest_round_from_contract(deploy_client,contract_abi,CONTRACT_ADDRESS); print(f">> Final confirmed round: {final_round}"); print(f">> Logs in: {log_dir}");
    print(">> Closing clients..."); [c.finish() for c in participant_clients if hasattr(c,'finish')]; print(">> Done.")

    # ========== 将每轮估计的并行耗时写入到 log 文件中 ==========
    estimated_durations_log_path = os.path.join(log_dir, "estimated_round_durations.log")
    with open(estimated_durations_log_path, 'a') as f:
        f.write("\n>>Estimated Per-Round Parallel Durations:\n")
        for round_num, duration in round_estimated_durations.items():
            f.write(f">>Round {round_num}: {duration:.4f} seconds\n")
    print(f"\n>>Estimated per-round parallel durations have been written to: {estimated_durations_log_path}")