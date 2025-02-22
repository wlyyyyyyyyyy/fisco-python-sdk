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
# import multiprocessing #  移除 multiprocessing 模块

# ----- 导入 PyTorch 库 -----
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io
import base64

# ----- 合约信息 (保持不变) -----
CONTRACT_NAME = "EvenSimplerFederatedLearning"
CONTRACT_PATH = "./contracts/EvenSimplerFederatedLearning.sol"
ABI_PATH = "./contracts/EvenSimplerFederatedLearning.abi"
BIN_PATH = "./contracts/EvenSimplerFederatedLearning.bin"
CONTRACT_NOTE_NAME = "federatedlearning_single_demo"
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

# ========== 加载 MNIST 训练数据集 (保持不变) ==========
def load_mnist_data_partition(batch_size=64):
    print("\n>>Loading MNIST Training Data ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("\n>>MNIST Training Data Loaded.")
    return train_loader

# ========== 加载 MNIST 测试数据集 (保持不变) ==========
def load_mnist_test_data(batch_size=64):
    print("\n>>Loading MNIST Test Data ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("\n>>MNIST Test Data Loaded.")
    return test_loader


# ========== 定义简单的 MLP 模型 (保持不变) ==========
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 classes for MNIST

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ========== 模型序列化函数 (保持不变) ==========
def serialize_model(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    model_str = base64.b64encode(model_bytes).decode()
    return model_str

# ========== 模型反序列化函数 (修改后，处理初始模型字符串) ==========
def deserialize_model(model_str):
    initial_model_placeholder = "Initial Model - Very Simple" #  定义初始模型占位符字符串
    if model_str == initial_model_placeholder: #  判断是否是初始模型字符串
        print("\n>>Initial Model Placeholder String Detected. Initializing new model...") # 打印提示信息
        model = MLP() #  如果是初始模型，则直接创建一个新的 MLP 模型
        return model #  直接返回新的模型，不进行解码
    else: #  如果不是初始模型字符串，则进行 Base64 解码和模型加载 (之前的逻辑)
        model = MLP()
        model_bytes = base64.b64decode(model_str.encode())
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        return model

# ========== 模型训练函数 (保持不变) ==========
def train_model(model, train_loader, epochs=1): # 固定 epochs=1
    print("\n>>Starting Local Model Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs): # 固定 epochs=1
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        print("\n>>Local Model Training Finished!")
        return model

# ========== 模型评估函数 (保持不变) ==========
def evaluate_model(model, test_loader):
    print("\n>>Starting Model Evaluation on Test Data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'\n>>Accuracy on Test Data: {accuracy:.2f}%')
    print("\n>>Model Evaluation Finished!")
    return accuracy


# ========== 模型上传函数 (保持不变) ==========
def upload_model_update(client, contract_abi, contract_address, updated_model_str, role_name):
    print(f"\n>>Uploading Model Update (from Role: {role_name})...")
    to_address = contract_address
    fn_name = "updateModel"
    args = [updated_model_str]
    receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args)
    if receipt is not None and receipt["status"] == 0:
        print(f"Upload Success (from Role: {role_name}), receipt: {receipt}")
        return True
    else:
        print(f"Upload Failed (from Role: {role_name}), receipt: {receipt}")
        return False

# ========== 模型下载函数 (保持不变) ==========
def download_global_model(client, contract_abi, contract_address, participant_address, role_name):
    print(f"\n>>Downloading Global Model (for Role: {role_name})...")
    to_address = contract_address
    fn_name = "getModel"
    get_result = client.call(to_address, contract_abi, fn_name, [])
    if get_result is not None:
        downloaded_model_str = get_result[0]
        print(f"Downloaded Model (Serialized, first 50 chars, for Role: {role_name}): {downloaded_model_str[:50]}...")
        return downloaded_model_str
    else:
        print(f"Download Failed (for Role: {role_name}), result: {get_result}")
        return None

# ==========  中央服务器角色函数 (模型测试) -  保持不变，但会顺序执行 ==========
def run_central_server(central_server_client, contract_abi, contract_address): # 移除 rounds 参数
    print(f"\n>>Starting Central Server (Testing Node)...")

    test_loader = load_mnist_test_data() # 加载测试数据集
    round_num = 1 # 固定轮数为 1

    print(f"\n>>>>> Federated Learning Round: {round_num} (Central Server - Testing) <<<<<<")
    print(f"\n>>Central Server (Testing Node) in Round: {round_num}")

    # -----  1.  模型下载 -----
    print("\n>>Central Server (Testing Node): Downloading Global Model...")
    downloaded_model_str = download_global_model(central_server_client, contract_abi, contract_address, None, "server")

    if downloaded_model_str:
        print("\n>>Global Model downloaded successfully.")
        global_model = deserialize_model(downloaded_model_str)
    else:
        print("\n>>Failed to download Global Model.")
        print("\n>>Evaluation Failed: No Global Model Downloaded.")
        return

    # -----  2. 模型评估 -----
    print("\n>>Central Server (Testing Node): Evaluating Global Model...")
    evaluate_model(global_model, test_loader) #  <--- 评估操作在下载之后
    print("\n>>Model Evaluation Completed.")

    print(f"\n>>Central Server (Testing Node) Completed!")


# ==========  参与者节点角色函数 (训练和上传) - 保持不变，但会顺序执行 ==========
def run_participant_node(participant_client, contract_abi, contract_address): # 移除 epochs 参数
    participant_id = "participant1" # 固定 participant_id
    epochs = 1 # 固定 epochs = 1
    print(f"\n>>Starting Participant Node (Training Node)...")

    # ----- 1. 下载全局模型 (Placeholder - 暂时跳过第一轮下载) -----
    downloaded_model_str = download_global_model(participant_client, contract_abi, contract_address, None, participant_id)
    if downloaded_model_str:
        print(f"\n>>Global Model downloaded successfully.")
        model = deserialize_model(downloaded_model_str)
    else:
        print(f"\n>>Failed to download Global Model. Using local initial model.")
        model = MLP()

    # ----- 2. 本地模型训练 -----
    train_loader = load_mnist_data_partition()
    trained_model = train_model(model, train_loader, epochs=epochs)
    updated_model_str = serialize_model(trained_model)

    # ----- 3. 上传模型更新 -----
    if upload_model_update(participant_client, contract_abi, contract_address, updated_model_str, participant_id):
        print(f"\n>>Model update uploaded successfully.")
    else:
        print(f"\n>>Model update upload failed.")

    print(f"\n>>Participant Node (Training Node) Finished Round.")


# ==========  主程序入口 (简化 Demo 模式 -  单进程双客户端) -  **修改后，再次运行中央节点进行评估** ==========
if __name__ == "__main__":
    print("\n>>Starting Federated Learning Demo (Minimal Single Script):----------------------------------------------------------")

    role = 'demo' #  固定为 demo 模式
    print(f"\n>>Running in DEMO mode (Minimal Dual Client Simulation - Single Process)") # 修改打印信息

    central_server_client = Bcos3Client() #  创建中央服务器客户端实例

    # ========== 部署合约 (Demo Mode - Server Part) ==========
    if not CONTRACT_ADDRESS:
        print("\n>>Demo Mode - Central Server (Testing Node): Deploying contract...")
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = central_server_client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"]
        print(f"Demo Mode - Central Server (Testing Node): Deploy contract success, contract address: {CONTRACT_ADDRESS}")
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
    else:
        print(f"\n>>Demo Mode - Central Server (Testing Node): Using existing contract at: {CONTRACT_ADDRESS}")

    # ========== 第一次运行中央服务器角色逻辑 (Demo Mode - Server Part - Initial Testing) ==========
    print("\n>>First Run: Central Server (Testing Node) - Initial Model Evaluation...") #  添加打印信息
    run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS) #  顺序执行中央服务器逻辑 (Initial Evaluation)
    print("\n>>First Run: Demo Mode - Central Server (Testing Node) process finished.") # 修改打印信息


    # ========== 模拟参与者节点角色 (Demo Mode - Participant Part - Training and Upload) ==========
    participant_client = Bcos3Client() #  创建参与者节点客户端实例
    run_participant_node(participant_client, contract_abi, CONTRACT_ADDRESS) #  顺序执行参与者节点逻辑 (Training and Upload)
    print("\n>>Demo Mode: Participant Node (Training Node) process finished.") # 修改打印信息

    # ========== 第二次运行中央服务器角色逻辑 (Demo Mode - Server Part - Re-evaluation after Training) ==========
    print("\n>>Second Run: Central Server (Testing Node) - Re-evaluating Updated Model...") #  添加打印信息
    run_central_server(central_server_client, contract_abi, CONTRACT_ADDRESS) #  **再次顺序执行中央服务器逻辑 (Re-evaluation)**
    print("\n>>Second Run: Demo Mode - Central Server (Testing Node) process finished.") # 修改打印信息


    print("\n>>Federated Learning Demo (Minimal Dual Client Simulation - Single Process) Finished!")