#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./myclient")

from client.stattool import StatTool
from bcos3sdk.bcos3client import Bcos3Client
from client.contractnote import ContractNote
from client.bcosclient import BcosClient
import os
from eth_utils import to_checksum_address
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client.bcoserror import BcosException, BcosError
from client_config import client_config as ClientConfig
import myclient.fl_1node_utils as fl_1node_utils  #  <--- 修改为导入 myclient.fl_utils 并使用别名

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
import datetime #  <--- 导入 datetime 模块，用于生成时间戳

# -----  参数解析  -----
args = fl_1node_utils.parse_arguments()  # 调用 fl_utils.py 中的参数解析函数

# ----- 配置参数 -----
demo_config = ClientConfig()  # 中央节点的配置使用默认 client_config

CONTRACT_NAME = "FederatedLearning"
CONTRACT_PATH = "./contracts/FederatedLearning.sol"
ABI_PATH = "./contracts/FederatedLearning.abi"
BIN_PATH = "./contracts/FederatedLearning.bin"
CONTRACT_NOTE_NAME = "federated_learning_demo"
CONTRACT_ADDRESS = ""

# 使用 fl_utils.py 中的 load_keystore_to_dict 加载 Keystore 文件
CENTRAL_NODE_ADDRESS = fl_1node_utils.load_keystore_to_dict("./bin/accounts/client0.keystore").get("address")
TRAINER_NODE_ADDRESSES = [
    fl_1node_utils.load_keystore_to_dict("./bin/accounts/client1.keystore").get("address"),
    fl_1node_utils.load_keystore_to_dict("./bin/accounts/client2.keystore").get("address"),
    fl_1node_utils.load_keystore_to_dict("./bin/accounts/client3.keystore").get("address"),
]

# 模型和训练参数，使用 args 对象获取参数值
MODEL_TYPE = args.model_type
DATASET_NAME = args.dataset
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs # 全局 epochs 参数 (用于合约交互，实际本地训练轮数使用 local_epochs)
NUM_ROUNDS = args.num_rounds
LOCAL_EPOCHS = args.local_epochs #  <--- 获取 local_epochs 参数
BATCHES_PER_ROUND = args.batches_per_round # <--- 获取 batches_per_round 参数

# ----- 日志配置 -----
LOG_DIR = "./myclient/fl_log" #  <---  日志文件夹路径
if not os.path.exists(LOG_DIR): #  <---  如果日志文件夹不存在，则创建
    os.makedirs(LOG_DIR)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #  <---  生成时间戳
LOG_FILE = os.path.join(LOG_DIR, f"fl_log_{TIMESTAMP}.log") #  <---  日志文件完整路径

def log_info(message): #  <---  日志记录函数
    print(message) #  仍然在终端输出一份
    with open(LOG_FILE, 'a') as f: #  追加写入日志文件
        f.write(message + "\n")

# ----- 模型定义 -----
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):  # 修改后的 CNN 模型，适用于 CIFAR-10 和 MNIST
    def __init__(self, num_channels):  # num_channels 参数控制输入通道数
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)  # 输入通道数可变
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 新增一个卷积层
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 全连接层输入尺寸需要根据新的特征图尺寸调整 (修正为 8x8)  <--- 修改为 8x8
        self.fc2 = nn.Linear(128, 10)  # 输出层保持不变 (假设都是 10 分类任务)

    def forward(self, x):
        # print(f"Input x shape in CNN forward: {x.shape}") #  <---  注释掉调试输出
        x = F.relu(self.conv1(x))
        # print(f"Shape after conv1: {x.shape}") #  <---  注释掉调试输出
        x = self.pool(x)
        # print(f"Shape after pool1: {x.shape}") #  <---  注释掉调试输出
        x = F.relu(self.conv2(x))
        # print(f"Shape after conv2: {x.shape}") #  <---  注释掉调试输出
        x = self.pool(x)
        # print(f"Shape after pool2: {x.shape}") #  <---  注释掉调试输出
        # 使用正确的特征维度计算: 64 * 8 * 8  <--- 修改为 8x8
        x = x.view(-1, 64 * 8 * 8)  #  修正 view 操作，使用 8*8 以匹配实际特征图尺寸  <--- 修改为 8x8
        # print(f"Shape after view (flatten): {x.shape}") #  <---  注释掉调试输出
        x = F.relu(self.fc1(x))
        # print(f"Shape after fc1: {x.shape}") #  <---  注释掉调试输出
        x = self.fc2(x)
        # print(f"Output shape in CNN forward: {x.shape}") #  <---  注释掉调试输出
        return x


# ----- 自动编译合约 -----
if os.path.isfile(demo_config.solc_path) or os.path.isfile(demo_config.solcjs_path):
       Compiler.compile_file(CONTRACT_PATH)
abi_file = ABI_PATH
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi


# ----- 模型训练 -----
def train_model(model, train_loader, local_epochs, batches_per_round): #  <--- 修改函数定义，新增 batches_per_round 参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    log_info(f"  train_loader batch size: {train_loader.batch_size}") #  <---  添加到日志
    log_info(f"  Number of batches in train_loader: {len(train_loader)}") #  <---  添加到日志

    # 使用传入的 local_epochs 参数控制本地训练轮数
    for epoch in range(local_epochs): #  <---  使用 local_epochs 参数
        batch_count = 0 #  <---  添加 batch 计数器
        for i, (images, labels) in enumerate(train_loader):
            if images.size(0) != labels.size(0):
                log_info(f"  Skipping batch index: {i} due to batch size mismatch.") #  <---  添加到日志
                log_info(f"  Images batch size: {images.size(0)}, Labels batch size: {labels.size(0)}") #  <---  添加到日志
                continue

            outputs = model(images)

            # ----- CRITICAL DEBUGGING PRINT STATEMENTS: ADD THESE EXACTLY AS SHOWN -----
            # print(f"Batch index: {i}, Images shape: {images.shape}, Labels shape: {labels.shape}") #  <---  注释掉调试输出
            # print(f"Outputs shape BEFORE criterion: {outputs.shape}") #  <---  注释掉调试输出

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                log_info (f'  Epoch [{epoch+1}/{local_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}') #  修改 epoch 打印，并添加到日志

            batch_count += 1 #  <---  batch 计数器加 1
            if batches_per_round != -1 and batch_count >= batches_per_round: #  <---  判断是否达到 batch 数量限制
                log_info(f"  達到每輪 batch 數量限制 ({batches_per_round}), 提前結束本輪訓練") #  打印提前结束训练的信息, 并添加到日志
                break #  <---  提前跳出 batch 循环

    return model


# ----- 模型评估 -----
def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 评估模式下不计算梯度
        correct = 0
        total = 0
        loss_sum = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')  # 使用 sum reduction 获取总 loss

        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()  # 累加 loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = loss_sum / total  # 计算平均 loss

        log_info(f'  Accuracy of the model on the test images: {accuracy:.2f} %') #  <---  添加到日志
        log_info(f'  Average Loss on the test images: {avg_loss:.4f}') #  <---  添加到日志
    model.train()  # 评估完成后，将模型设置回训练模式 (如果你后面还需要继续训练)
    return accuracy, avg_loss


# ----- 模型哈希生成 -----
def generate_model_hash(model):
    model_state = model.state_dict()
    # 排序字典的键
    model_bytes = json.dumps(model_state, sort_keys=True,
                              default=lambda o: o.__dict__ if isinstance(o, torch.Tensor) else o).encode('utf-8')
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    return model_hash


# bytes32 常量
ROLE_CENTRAL = hashlib.sha256("CENTRAL".encode('utf-8')).hexdigest()
ROLE_TRAINER = hashlib.sha256("TRAINER".encode('utf-8')).hexdigest()

try:
    stat = StatTool.begin()
    client = Bcos3Client(ClientConfig())  # Central client (client0) 使用默认配置对象
    log_info(client.getinfo()) #  <---  添加到日志

    # ========== 部署合约 ==========
    if not CONTRACT_ADDRESS:
        log_info("\n>>Deploy:----------------------------------------------------------") #  <---  添加到日志
        with open(BIN_PATH, 'r') as load_f:
            contract_bin = load_f.read()
            load_f.close()
        deploy_result = client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            log_info("deploy contract failed, receipt:{}".format(deploy_result)) #  <---  添加到日志
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"]
        log_info("deploy success, receipt:{}".format(deploy_result)) #  <---  添加到日志
        log_info("new address : {}".format(CONTRACT_ADDRESS)) #  <---  添加到日志
        contract_name = CONTRACT_NAME
        log_info("contract name: {}".format(contract_name)) #  <---  添加到日志
        memo = "tx:" + deploy_result["transactionHash"]
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, contract_name, CONTRACT_ADDRESS)
    else:
        log_info(f"\n>>Use existed contract at: {CONTRACT_ADDRESS}") #  <---  添加到日志

    # ========== 合约初始化 ==========
    log_info("\n>>Contract Initialize:----------------------------------------------------------") #  <---  添加到日志
    to_address = CONTRACT_ADDRESS
    fn_name = "initialize"
    args_init = [to_checksum_address(CENTRAL_NODE_ADDRESS)]
    receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args_init)
    log_info("receipt:{}".format(receipt)) #  <---  添加到日志

    # ========== 注册训练节点 ==========
    log_info("\n>>Register Trainer Nodes:----------------------------------------------------------") #  <---  添加到日志
    trainer_clients = []
    for i, trainer_address in enumerate(TRAINER_NODE_ADDRESSES):
        config_file_path = "client_config_trainer{}.toml".format(i + 1)  # 保留 TOML 文件，用于客户端特定的设置 (虽然目前未使用)
        sdk_config_file_path = "./bcos3sdklib/bcos3_sdk_config_trainer{}.ini".format(i + 1)  # *SDK* 配置 INI 文件的路径
        trainer_client_config = ClientConfig(config_file=sdk_config_file_path)  # 创建 trainer 的 client_config *对象*，并传递 SDK 配置文件路径
        trainer_client = Bcos3Client(trainer_client_config)  # 将配置*对象* 传递给 Bcos3Client
        trainer_clients.append(trainer_client)
        to_address = CONTRACT_ADDRESS
        fn_name = "registerParticipant"
        args_register = [to_checksum_address(trainer_address), ROLE_TRAINER]
        receipt = trainer_client.sendRawTransaction(to_address, contract_abi, fn_name, args_register)  # 使用 trainer client 发送注册交易
        log_info(f"  Register Trainer Node {i + 1} receipt:{receipt}") #  <---  添加到日志

    # ========== 记录运行参数到日志 ==========
    log_info("\n>>运行参数:----------------------------------------------------------") #  <---  添加到日志
    log_info(f"  Model Type: {args.model_type}") #  <---  添加到日志
    log_info(f"  Dataset: {args.dataset}") #  <---  添加到日志
    log_info(f"  Batch Size: {args.batch_size}") #  <---  添加到日志
    log_info(f"  Epochs (Global): {args.epochs}") #  <---  添加到日志
    log_info(f"  Local Epochs: {args.local_epochs}") #  <---  添加到日志
    log_info(f"  Batches per Round: {args.batches_per_round}") #  <---  添加到日志
    log_info(f"  Number of Rounds: {args.num_rounds}") #  <---  添加到日志

    # ========== 多轮联邦学习循环 ==========
    for round_num in range(NUM_ROUNDS):
        log_info(f"\n========== Federated Learning Round: {round_num + 1}/{NUM_ROUNDS} ==========") #  <---  添加到日志

        # ========== 加载数据并训练模型 (每个训练节点) ==========
        if DATASET_NAME == "MNIST":
            train_loader = fl_1node_utils.load_mnist_data(BATCH_SIZE)  # 加载 MNIST 训练数据
            test_loader = fl_1node_utils.load_mnist_test_data(BATCH_SIZE)  # 加载 MNIST 测试数据
            num_channels = 1  # MNIST 灰度图像，通道数为 1
        elif DATASET_NAME == "CIFAR10":
            train_loader = fl_1node_utils.load_cifar10_data(BATCH_SIZE)  # 加载 CIFAR-10 训练数据
            test_loader = fl_1node_utils.load_cifar10_test_data(BATCH_SIZE)  # 加载 CIFAR-10 测试数据
            num_channels = 3  # CIFAR-10 彩色图像，通道数为 3
        else:
            log_info("Invalid Dataset Type!") #  <---  添加到日志
            exit()

        # 根据 MODEL_TYPE 选择模型
        if MODEL_TYPE == "MLP":
            model = MLP()
        elif MODEL_TYPE == "CNN":
            model = CNN(num_channels=num_channels)  # 将通道数传递给 CNN 模型
        else:
            log_info("Invalid Model Type!") #  <---  添加到日志
            exit()

        log_info(f"\n>>Train Model ({MODEL_TYPE}) by each Trainer Node (Round {round_num + 1}):-----------------------") #  <---  添加到日志
        trained_models = []
        model_hashes = []
        for i, trainer_client in enumerate(trainer_clients):
            log_info(f"\n  >>Trainer Node {i + 1} Training Model (Round {round_num + 1})...") #  <---  添加到日志
            # 使用 args.local_epochs 和 args.batches_per_round 参数控制本地训练
            trained_model = train_model(model, train_loader, local_epochs=args.local_epochs, batches_per_round=args.batches_per_round) #  <---  传递 args.local_epochs 和 batches_per_round
            trained_models.append(trained_model)
            model_hash = generate_model_hash(trained_model)
            model_hashes.append(model_hash)
            log_info(f"  Trainer Node {i + 1} Model Training Finished! (Round {round_num + 1})") #  <---  添加到日志
            log_info(f"  Trainer Node {i + 1} Generated Model Hash (Hex): {model_hash}") #  <---  添加到日志

        # ========== 上传模型更新 (每个训练节点) ==========
        log_info(f"\n>>Upload Model Update by each Trainer Node (Round {round_num + 1}):-----------------------") #  <---  添加到日志
        upload_receipts = []
        for i, trainer_client in enumerate(trainer_clients):
            to_address = CONTRACT_ADDRESS
            fn_name = "uploadModelUpdate"
            args_upload = [model_hashes[i]]
            receipt = trainer_client.sendRawTransaction(to_address, contract_abi, fn_name, args_upload)  # 使用 trainer client 上传模型更新
            upload_receipts.append(receipt)
            log_info(f"  Trainer Node {i + 1} Upload Model Update Receipt (Round {round_num + 1}): {receipt}") #  <---  添加到日志 (简化输出)

        # ========== 模拟模型聚合 (中央节点操作) ==========
        log_info(f"\n>>Aggregate Model (Central Node) (Round {round_num + 1}):----------------------------------") #  <---  添加到日志
        # 简化: 直接使用最后一个 Trainer 模型的哈希作为聚合模型哈希 (实际场景需要模型聚合算法)
        aggregated_model_hash = model_hashes[-1]

        to_address = CONTRACT_ADDRESS
        fn_name = "aggregateModel"
        args_aggregate = [aggregated_model_hash]
        receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args_aggregate)  # 中央节点执行聚合
        log_info(f"  Aggregate Model Receipt (Round {round_num + 1}): {receipt}") #  <---  添加到日志 (简化输出)

        # ========== 评估聚合模型 (中央节点操作) ==========
        log_info(f"\n>>Evaluate Aggregated Model (Central Node) (Round {round_num + 1}):-------------------------") #  <---  添加到日志
        # **注意**: 这里仍然使用最后一个训练的模型 trained_models[-1] 近似代表聚合模型进行评估
        accuracy, avg_loss = evaluate_model(trained_models[-1], test_loader)  # 评估最后一个训练的模型
        log_info(f"  Aggregated Model Evaluation (Round {round_num + 1}) - Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}") #  <---  添加到日志

    stat.done()
    reqcount = next(client.request_counter)
    log_info("\n>>运行结束统计:----------------------------------------------------------") #  <---  添加到日志
    log_info("  done,demo_tx,total request {},usedtime {},avgtime:{}".format( #  <---  添加到日志
        reqcount, stat.time_used, (stat.time_used / reqcount)
    ))

except BcosException as e:
    log_info("execute federated_learning_demo failed ,BcosException for: {}".format(e)) #  <---  添加到日志
    traceback.print_exc()
except BcosError as e:
    log_info("execute federated_learning_demo failed ,BcosError for: {}".format(e)) #  <---  添加到日志
    traceback.print_exc()
except Exception as e:
    if 'client' in locals():
        client.finish()
    traceback.print_exc()
finally:
    if 'client' in locals():
        client.finish()
    for trainer_client in trainer_clients:
        trainer_client.finish()
    sys.exit(0)
    





#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import json

# ----- data_utils.py 内容 (已合并到 fl_utils.py) -----

# ----- MNIST 数据加载和预处理 -----
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) #  明确设置 num_workers=0
    return train_loader

def load_mnist_test_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # 明确设置 num_workers=0
    return test_loader

# ----- CIFAR-10 数据加载和预处理 -----
def load_cifar10_data(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 的均值和标准差
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for debugging

    return train_loader

def load_cifar10_test_data(batch_size=64):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers=0 for debugging
    return test_loader


# ----- arg_parser.py 内容 (已合并到 fl_utils.py) -----

def load_keystore_to_dict(keystore_file):
    try:
        with open(keystore_file, 'r') as f:
            keystore_data = json.load(f)
            return keystore_data
    except Exception as e:
        print(f"Error reading keystore file: {e}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('--model_type', type=str, default='MLP', choices=['MLP', 'CNN'], help='Model type (MLP or CNN)')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'], help='Dataset to use (MNIST or CIFAR10)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs per round (Global Epochs, for contract interaction, not used for local training)') # 修改了 epochs 参数的描述
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs per round (for each trainer node)') # 新增 local_epochs 参数
    parser.add_argument('--batches_per_round', type=int, default=-1, help='Number of batches per round for local training (-1 means use all batches)') # 新增 batches_per_round 参数
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    print("Parsed arguments (from fl_utils.py test):") # 修改了打印信息，更清晰地表明是来自 fl_utils.py 的测试
    print(f"  Model Type: {args.model_type}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs (Global): {args.epochs}") # 修改了打印信息
    print(f"  Local Epochs: {args.local_epochs}") # 打印 local_epochs 参数
    print(f"  Number of Rounds: {args.num_rounds}")
    # 可以添加更多 fl_utils.py 的测试代码