#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
from client.stattool import StatTool
from client.contractnote import ContractNote
from client.bcosclient import BcosClient  # 导入 BcosClient (v2)
import os
from eth_utils import to_checksum_address
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client.bcoserror import BcosException, BcosError
from client_config import client_config

import traceback
import json
import hashlib
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from multiprocessing import Process, Manager, Event  # Import multiprocessing modules

def load_keystore_to_dict(keystore_file):
    try:
        with open(keystore_file, 'r') as f:
            keystore_data = json.load(f)
            return keystore_data
    except Exception as e:
        print(f"Error reading keystore file: {e}")
        return None


# ----- 配置参数 -----
demo_config = client_config

CONTRACT_NAME = "FederatedLearning"
CONTRACT_PATH = "./contracts/FederatedLearning.sol"
ABI_PATH = "./contracts/FederatedLearning.abi"
BIN_PATH = "./contracts/FederatedLearning.bin"
CONTRACT_NOTE_NAME = "federated_learning_demo"
CONTRACT_ADDRESS = ""

CENTRAL_NODE_ADDRESS_RAW = load_keystore_to_dict("./bin/accounts/client0.keystore").get("address") # Load raw address
print(f"Raw CENTRAL_NODE_ADDRESS from keystore: {CENTRAL_NODE_ADDRESS_RAW}") # Debug: Print raw address
CENTRAL_NODE_ADDRESS = to_checksum_address(CENTRAL_NODE_ADDRESS_RAW) # Convert to checksum address
print(f"Checksum CENTRAL_NODE_ADDRESS: {CENTRAL_NODE_ADDRESS}") # Debug: Print checksum address


# 训练节点配置 (这里模拟多个训练节点)
TRAINER_NODE_ADDRESSES = [
    load_keystore_to_dict("./bin/accounts/client1.keystore").get("address"),
    load_keystore_to_dict("./bin/accounts/client2.keystore").get("address"),
    load_keystore_to_dict("./bin/accounts/client3.keystore").get("address"),
    load_keystore_to_dict("./bin/accounts/client4.keystore").get("address"),
    load_keystore_to_dict("./bin/accounts/client5.keystore").get("address"),
]

# 模型类型选择: "MLP", "CNN"
MODEL_TYPE = "CNN"  #  选择一个模型类型

BATCH_SIZE = 64
EPOCHS = 1
MAX_ROUNDS = 3 # For multi-round FL simulation
QUERY_INTERVAL = 2 # Interval to query contract status


# ----- 模型定义 -----
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc(x)
        return x


# ----- 自动编译合约 -----
if os.path.isfile(demo_config.solc_path) or os.path.isfile(demo_config.solcjs_path):
    Compiler.compile_file(CONTRACT_PATH)
abi_file = ABI_PATH
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

# ----- MNIST 数据加载和预处理 -----
def load_mnist_data(num_trainers): # Modified to accept num_trainers
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    subset_size = len(train_dataset) // num_trainers
    train_loaders = []
    for i in range(num_trainers):
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        train_loaders.append(train_loader)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loaders, test_loader


# ----- 模型训练 -----
def train_model(model, train_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    return model

# ----- 模型评估 -----
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        return accuracy

# ----- 模型哈希生成 -----
def generate_model_hash(model):
    model_state = model.state_dict()
     # 排序字典的键
    model_bytes = json.dumps(model_state, sort_keys=True, default=lambda o: o.__dict__ if isinstance(o, torch.Tensor) else o).encode('utf-8')
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    return model_hash

# bytes32 常量
ROLE_CENTRAL = hashlib.sha256("CENTRAL".encode('utf-8')).hexdigest()
ROLE_TRAINER = hashlib.sha256("TRAINER".encode('utf-8')).hexdigest()


# ==========================
# 训练节点进程 (v2 版本 BcosClient)
# ==========================
def run_trainer_v2(node_id, model_type, train_loader, test_loader, contract_address, contract_abi, stop_event):
    """
    训练节点的工作流程 (v2 版本 BcosClient)
    """
    print(f"Trainer {node_id} starting (V2 Client)...") # 明确标识 v2 client
    try:
        # 每个训练节点创建一个 BcosClient 实例, 并指定私钥文件
        client = BcosClient(demo_config, account_keyfile=f"./bin/accounts/client{node_id}.keystore") # 使用 account_keyfile 参数

        # 选择模型
        if model_type == "MLP":
            model = MLP()
        elif model_type == "CNN":
            model = CNN()
        else:
            print("Invalid Model Type!")
            return

        # 主循环
        while not stop_event.is_set(): # 检查停止事件
            # 获取当前模型版本 (用于判断是否需要训练)
            try:
                current_version = client.call(contract_address, contract_abi, "currentModelVersion", []) # v2 版本 call 不需要 contract_abi
                current_version = int(current_version[0]) # v2 返回的是 list, 取第一个元素并转换为 int
            except Exception as e:
                print(f"Trainer {node_id} (V2 Client): Error getting currentModelVersion: {e}")
                time.sleep(QUERY_INTERVAL)
                continue

            # 模拟训练 (假设每个训练节点在每个版本都训练一次)
            print(f"Trainer {node_id} (V2 Client): Training model (version {current_version + 1})...")  # 从 current_version + 1 开始
            trained_model = train_model(model, train_loader)

            # 评估训练后的模型
            print(f"Trainer {node_id} (V2 Client): Evaluating trained model...")
            evaluate_model(trained_model, test_loader)

            # 上传模型更新
            print(f"Trainer {node_id} (V2 Client): Uploading model update...")
            model_hash = generate_model_hash(trained_model)

            receipt = client.sendRawTransaction(contract_address, contract_abi, "uploadModelUpdate", args=[model_hash]) # v2 sendRawTransaction 参数调整

            # 将模型更新为训练好的模型, 为下一轮训练做准备。
            model.load_state_dict(trained_model.state_dict())

            time.sleep(QUERY_INTERVAL)  # 等待一段时间

    except Exception as e:
        traceback.print_exc()
    finally:
        if 'client' in locals():  # 检查 client 是否已定义
            client.finish()
        print(f"Trainer {node_id} (V2 Client) exiting...")

# ==========================
# 中央节点/聚合节点进程 (v2 版本 BcosClient)
# ==========================
def run_sponsor_v2(contract_address, contract_abi, num_trainers, test_loader, stop_event):
    """
    中央节点/聚合节点的工作流程 (v2 版本 BcosClient)
    """
    print("Sponsor starting (V2 Client)...") # 明确标识 v2 client
    try:
        client = BcosClient(demo_config)  # 使用默认账户 (client0), v2 client 初始化方式

        # 获取已经训练过的模型
        if MODEL_TYPE == "MLP":
          model = MLP()
        elif MODEL_TYPE == "CNN":
          model = CNN()
        else:
          print("Invalid Model Type!")
          return

        round_num = 0
        while round_num < MAX_ROUNDS and not stop_event.is_set(): # 检查停止事件
            round_num += 1
            print(f"\n>>Starting Round {round_num} (V2 Client):----------------------------------------------------------")

            # 等待所有训练节点上传模型更新 (简化版，通过轮询实现)
            print("Sponsor (V2 Client): Waiting for trainers to upload updates...")
            while True:
                try:
                    current_version = client.call(contract_address, contract_abi, "currentModelVersion", []) # v2 版本 call 不需要 contract_abi
                    current_version = int(current_version[0]) # v2 返回的是 list, 取第一个元素并转换为 int
                    uploaded_count = 0
                    for i in range(num_trainers):
                        try:
                            result = client.call(contract_address, contract_abi, "modelUpdates", [current_version, TRAINER_NODE_ADDRESSES[i]], []) # v2 版本 call 参数调整
                            if result[0] != '0x0000000000000000000000000000000000000000000000000000000000000000':
                                uploaded_count += 1
                        except Exception as e:
                            pass
                    if uploaded_count == num_trainers:
                        print(f"Sponsor (V2 Client): All {num_trainers} trainers have uploaded updates for version {current_version}.")
                        break
                    else:
                        print(f"Sponsor (V2 Client): Waiting for trainers... ({uploaded_count}/{num_trainers})")
                        time.sleep(QUERY_INTERVAL)

                except Exception as e:
                    print(f"Sponsor (V2 Client): Error while waiting for trainers: {e}")
                    time.sleep(QUERY_INTERVAL)
                    continue


            # 模拟模型聚合 (简化版，选择最后一个上传的模型的哈希)
            print("Sponsor (V2 Client): Aggregating model...")
            # 找到最新上传的那个模型
            latest_model_hash = None
            for trainer_address in TRAINER_NODE_ADDRESSES:
                try:
                    update = client.call(contract_address, contract_abi, "modelUpdates", [current_version, trainer_address], []) # v2 版本 call 参数调整
                    if update[0] != "0x0000000000000000000000000000000000000000000000000000000000000000":
                        latest_model_hash = update[0] #获取到最新的
                except Exception as e: #忽略可能的错误
                    pass
            aggregated_model_hash = latest_model_hash

            # 触发模型聚合 (更新 currentModelVersion)
            receipt = client.sendRawTransaction(contract_address, contract_abi, "aggregateModel", args=[aggregated_model_hash]) # v2 sendRawTransaction 参数调整

            # 评估聚合模型 (简化版，直接加载最后一个上传的训练节点的模型)
            print("Sponsor (V2 Client): Evaluating aggregated model (simplified)...")  # 简化评估
            if latest_model_hash:
                # 真实场景: 你需要根据 latest_model_hash, 从链下存储 (如 IPFS) 中下载模型
                print("Sponsor (V2 Client): (Simplified) 'Loading' aggregated model for evaluation...")
                evaluate_model(model, test_loader)  # 使用测试集评估模型

            time.sleep(QUERY_INTERVAL) #等待

    except Exception as e:
        traceback.print_exc()
    finally:
        client.finish()
        print("Sponsor (V2 Client) exiting...")


# ==========================
# 主程序入口 (v2 版本 BcosClient)
# ==========================
if __name__ == "__main__":

    # 解析合约 ABI (bcos3sdk 需要在部署前知道 abi, v2 版本 client 也需要 abi)
    # 自动编译合约
    if os.path.isfile(demo_config.solc_path) or os.path.isfile(demo_config.solcjs_path):
      Compiler.compile_file(CONTRACT_PATH, output_path="./contracts")

    abi_file = ABI_PATH
    data_parser = DatatypeParser()
    data_parser.load_abi_file(abi_file)
    contract_abi = data_parser.contract_abi

    # 加载数据
    num_trainers = len(TRAINER_NODE_ADDRESSES)
    train_loaders, test_loader = load_mnist_data(num_trainers) # Pass num_trainers to load_mnist_data

    # 获取合约地址 (部署或从 contractnote 读取)
    try:
        client = BcosClient(demo_config) # 使用 BcosClient (v2) 初始化客户端

        if not CONTRACT_ADDRESS:
            print("\n>>Deploy:----------------------------------------------------------")
            with open(BIN_PATH, 'r') as f:
                contract_bin = f.read()
                f.close()
            # v2 版本 deploy 第一个参数是 abi 文件路径， 第二个参数是 bin 文件路径
            receipt = client.deploy(ABI_PATH, BIN_PATH)
            if receipt is None:  #deploy 接口现在可能返回 None
                print("deploy contract failed, receipt is None")
                exit(1)

            if receipt['status'] != 0: # 检查 status, v2 版本 receipt 是 dict
                print("deploy contract failed, receipt:", receipt)
                exit(1)

            CONTRACT_ADDRESS = receipt["contractAddress"]
            print("deploy success, receipt:", receipt)
            print("new address : ", CONTRACT_ADDRESS)

        else:
            print(f"\n>>Use existed contract at: {CONTRACT_ADDRESS}")

        # 初始化合约 (只需要由中央节点/部署者执行一次)
        print("\n>>Contract Initialize:----------------------------------------------------------")
        receipt = client.sendRawTransaction(CONTRACT_ADDRESS, contract_abi, "initialize", args=[CENTRAL_NODE_ADDRESS]) # v2 sendRawTransaction 参数调整
        print("receipt:", receipt)

        # 注册训练节点
        print("\n>>Register Trainer Nodes:----------------------------------------------------------")
        for trainer_address in TRAINER_NODE_ADDRESSES: # Loop through trainer addresses
            receipt = client.sendRawTransaction(CONTRACT_ADDRESS, contract_abi, "registerParticipant", args=[to_checksum_address(trainer_address), ROLE_TRAINER]) # v2 sendRawTransaction 参数调整
            print(f"Register Trainer Node {trainer_address} receipt:", receipt)

        # 获取训练节点数量
        trainers_num_onchain = client.call(CONTRACT_ADDRESS, contract_abi, "getTrainingNodeCount", []) # v2 版本 call 不需要 contract_abi, 参数调整
        trainers_num_onchain = int(trainers_num_onchain[0]) # v2 返回的是 list, 取第一个元素并转换为 int
        print(f"trainers_num on chain: {trainers_num_onchain}")


    except Exception as e:
        print("Contract deployment/initialization failed.")
        traceback.print_exc()
        exit()
    finally:
        client.finish()

    # 使用 Manager 创建共享事件
    with Manager() as manager:
        stop_event = manager.Event()  # 创建一个共享事件

        # 创建进程
        processes = []

        # 创建训练节点进程 (使用 v2 版本 client 的 trainer 进程)
        for i in range(num_trainers):
            train_loader_for_trainer = train_loaders[i] # Get specific train_loader for each trainer
            p = Process(target=run_trainer_v2, args=(i + 1, MODEL_TYPE, train_loader_for_trainer, test_loader, CONTRACT_ADDRESS, contract_abi, stop_event)) #  使用 run_trainer_v2
            processes.append(p)
            p.start()

        # 创建中央节点/聚合节点进程 (使用 v2 版本 client 的 sponsor 进程)
        p = Process(target=run_sponsor_v2, args=(CONTRACT_ADDRESS, contract_abi, num_trainers, test_loader, stop_event)) # 使用 run_sponsor_v2
        processes.append(p)
        p.start()

        # 等待一段时间 (例如，等待训练完成)
        time.sleep(60)  # 等待 60 秒 (根据需要调整)

        # 设置停止事件
        stop_event.set()
        print("Stop event set. Waiting for processes to finish...")

        # 等待所有进程结束
        for p in processes:
            p.join()

    print("All processes finished.")