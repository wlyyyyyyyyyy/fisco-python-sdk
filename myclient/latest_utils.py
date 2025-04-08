# --- START OF FILE latest_utils.py ---
# latest_utils.py (Decentralized Version - On-Chain Hash Calculation, Contract: Latest - Formatted)
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import io
import base64
import os
import json
import time
import traceback
import copy

# --- No external hash libraries needed or imported ---
print("Using On-Chain Hash Calculation (no external eth-hash/eth-abi needed).")

# ========== 操作日志记录函数 ==========
def log_operation(log_dir, round_num, role_name, operation_type, message):
    """记录操作日志到文件和控制台"""
    log_filename = os.path.join(log_dir, f"operations_log.txt")
    os.makedirs(log_dir, exist_ok=True) # 确保目录存在
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_info = f'[{timestamp}] Round: {round_num}, Role: {role_name}, Op: {operation_type}, Msg: {message}'
    print(log_info) # 同时打印到控制台
    try:
        # 追加模式写入日志文件
        with open(log_filename, 'a') as log_file:
            log_file.write(log_info + '\n')
    except Exception as e:
        print(f"[ERROR] Failed writing log to {log_filename}: {e}")

# --- In latest_utils.py ---

# ========== 数据加载函数 (使用调整后的结构) ==========

def load_mnist_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3):
    """加载并划分 MNIST 训练数据 (调整后结构)"""
    print(f"\n>> Loading MNIST Training Data for P:{participant_id}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    except Exception as e:
        print(f"[ERROR] MNIST load failed: {e}")
        return None

    num_samples = len(full_train_dataset)
    if total_participants <= 0:
        print("[ERROR] total_participants must be > 0")
        return None

    # --- 1. 验证 participant_id 类型和格式 ---
    print(f"DEBUG [load_mnist]: Type: {type(participant_id)}, Value: {repr(participant_id)}")
    if not isinstance(participant_id, str):
        print(f"[ERROR] [load_mnist]: Expected participant_id to be str, got {type(participant_id)}")
        return None

    try:
        # 尝试从 participant_id 解析索引
        participant_index = int(participant_id.replace("participant", "")) - 1
    except ValueError:
        # 如果格式不是 "participantX"，解析失败
        print(f"[ERROR] [load_mnist]: Invalid participant_id format for int conversion: {participant_id}. Expected 'participantX'.")
        return None

    # --- 2. 验证解析出的索引是否在有效范围内 ---
    if not (0 <= participant_index < total_participants):
        print(f"[ERROR] [load_mnist]: Invalid participant index {participant_index} (must be between 0 and {total_participants - 1}) derived from {participant_id}.")
        return None

    # --- 3. 如果 participant_id 有效，再进行分区计算 ---
    partition_size = num_samples // total_participants
    start_index = participant_index * partition_size
    end_index = (start_index + partition_size) if participant_index != total_participants - 1 else num_samples

    indices = list(range(start_index, end_index))
    if not indices:
        print(f"[WARN] [load_mnist]: No data samples assigned to P:{participant_id} (Indices {start_index} to {end_index})")
        return None

    # --- 4. 创建数据集和加载器 ---
    partitioned_dataset = Subset(full_train_dataset, indices)
    train_loader = DataLoader(partitioned_dataset, batch_size=batch_size, shuffle=True)
    print(f">> MNIST Data Loaded for P:{participant_id} ({len(partitioned_dataset)} samples).")
    return train_loader

def load_cifar10_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3):
    """加载并划分 CIFAR-10 训练数据 (调整后结构)"""
    print(f"\n>> Loading CIFAR-10 Training Data for P:{participant_id}...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    except Exception as e:
        print(f"[ERROR] CIFAR10 load failed: {e}")
        return None

    num_samples = len(full_train_dataset)
    if total_participants <= 0:
        print("[ERROR] total_participants must be > 0")
        return None

    # --- 1. 验证 participant_id 类型和格式 ---
    print(f"DEBUG [load_cifar10]: Type: {type(participant_id)}, Value: {repr(participant_id)}")
    if not isinstance(participant_id, str):
        print(f"[ERROR] [load_cifar10]: Expected participant_id to be str, got {type(participant_id)}")
        return None

    try:
        # 尝试从 participant_id 解析索引
        participant_index = int(participant_id.replace("participant", "")) - 1
    except ValueError:
        # 如果格式不是 "participantX"，解析失败
        print(f"[ERROR] [load_cifar10]: Invalid participant_id format for int conversion: {participant_id}. Expected 'participantX'.")
        return None

    # --- 2. 验证解析出的索引是否在有效范围内 ---
    if not (0 <= participant_index < total_participants):
        print(f"[ERROR] [load_cifar10]: Invalid participant index {participant_index} (must be between 0 and {total_participants - 1}) derived from {participant_id}.")
        return None

    # --- 3. 如果 participant_id 有效，再进行分区计算 ---
    partition_size = num_samples // total_participants
    start_index = participant_index * partition_size
    end_index = (start_index + partition_size) if participant_index != total_participants - 1 else num_samples

    indices = list(range(start_index, end_index))
    if not indices:
        print(f"[WARN] [load_cifar10]: No data samples assigned to P:{participant_id} (Indices {start_index} to {end_index})")
        return None

    # --- 4. 创建数据集和加载器 ---
    partitioned_dataset = Subset(full_train_dataset, indices)
    train_loader = DataLoader(partitioned_dataset, batch_size=batch_size, shuffle=True)
    print(f">> CIFAR-10 Data Loaded for P:{participant_id} ({len(partitioned_dataset)} samples).")
    return train_loader

# --- (latest_utils.py 文件中的其他函数保持不变) ---

def load_mnist_test_data(batch_size=1000):
    """加载完整的 MNIST 测试数据"""
    print("\n>> Loading MNIST Test Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f">> MNIST Test Loaded ({len(test_dataset)} samples).")
        return test_loader
    except Exception as e:
        print(f"[ERROR] MNIST test load failed: {e}")
        return None

def load_cifar10_test_data(batch_size=1000):
    """加载完整的 CIFAR-10 测试数据"""
    print("\n>> Loading CIFAR-10 Test Data...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f">> CIFAR-10 Test Loaded ({len(test_dataset)} samples).")
        return test_loader
    except Exception as e:
        print(f"[ERROR] CIFAR10 test load failed: {e}")
        return None

# ========== 模型定义 ==========
class MLP(nn.Module):
    """简单的多层感知机模型 (适用于 MNIST)"""
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    """简单的卷积神经网络模型 (适用于 CIFAR-10)"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes) # Calculated for 32x32 input

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = self.fc(x)
        return x

# ========== 模型序列化/反序列化 ==========
def serialize_model(model):
    """将 PyTorch 模型的状态字典序列化为 base64 字符串"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    model_str = base64.b64encode(model_bytes).decode('utf-8')
    return model_str

def deserialize_model(model_str, model_type='cnn', dataset='cifar10'):
    """将 base64 字符串反序列化回 PyTorch 模型"""
    if not model_str:
        print("[ERROR] deserialize_model received empty string.")
        return None
    try:
        model_bytes = base64.b64decode(model_str.encode('utf-8'))
        buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(buffer)

        # 根据类型创建新模型实例
        if model_type == 'mlp':
            model = MLP()
        elif model_type == 'cnn':
            model = CNN(num_classes=10) # Assume 10 classes
        else:
            raise ValueError(f"Unsupported model type for deserialization: {model_type}")

        model.load_state_dict(state_dict)
        return model
    except base64.binascii.Error as e:
         print(f"[ERROR] Base64 decode failed: {e} (String start: {model_str[:50]}...)")
         return None
    except Exception as e:
        print(f"[ERROR] Model deserialization failed: {e}")
        traceback.print_exc()
        return None

# ========== 模型训练函数 ==========
def train_model(model, train_loader, epochs=1, log_dir="fl_log", round_num=1, participant_id="participant1"):
    """在本地数据上训练模型"""
    if train_loader is None:
        print(f"[WARN] No training data for P:{participant_id} R:{round_num}. Skipping.")
        return model # Return original model if no data

    print(f"\n>> Starting Training P:{participant_id} R:{round_num} E:{epochs}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")
    model.to(device) # 将模型移动到设备
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 定义优化器

    log_filename = os.path.join(log_dir, f"train_log_round_{round_num}_{participant_id}.txt")
    os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在

    model.train() # 设置模型为训练模式
    last_epoch_avg_loss = 0.0 # 用于记录最终的平均损失

    try:
        with open(log_filename, 'a') as log_file:
            log_file.write(f"--- Train Start R:{round_num} P:{participant_id} E:{epochs} T:{time.strftime('%H:%M:%S')} ---\n")
            # 训练指定轮数
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                # 遍历训练数据加载器
                for batch_idx, (data, target) in enumerate(train_loader):
                    # 将数据移动到设备
                    data, target = data.to(device), target.to(device)
                    # 清零梯度
                    optimizer.zero_grad()
                    # 前向传播
                    output = model(data)
                    # 计算损失
                    loss = criterion(output, target)
                    # 反向传播
                    loss.backward()
                    # 更新参数
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_batches += 1

                    # 每隔一定批次打印一次日志
                    if batch_idx % 100 == 0:
                        log_info = f' P:{participant_id} R:{round_num} E:{epoch+1}/{epochs} B:{batch_idx}/{len(train_loader)} Loss:{loss.item():.4f}'
                        print(log_info)
                        log_file.write(log_info + '\n')

                # 计算并记录该 epoch 的平均损失
                last_epoch_avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
                print(f'>> P:{participant_id} R:{round_num} Epoch {epoch+1} Average Loss: {last_epoch_avg_loss:.4f}')
                log_file.write(f'Epoch {epoch+1} Average Loss: {last_epoch_avg_loss:.4f}\n')

            # 训练完成日志
            print(f">> Training Finished P:{participant_id} R:{round_num}! Last Epoch Avg Loss: {last_epoch_avg_loss:.4f}")
            log_file.write(f"--- Train End Last Epoch Avg Loss: {last_epoch_avg_loss:.4f} T:{time.strftime('%H:%M:%S')} ---\n")

    except Exception as e:
        print(f"[ERROR] Training failed for P:{participant_id} R:{round_num}: {e}")
        traceback.print_exc()
        return model # 发生错误时返回原始模型，避免后续出错

    return model # 返回训练后的模型对象

# ========== 模型聚合函数 ==========
def aggregate_global_model(participant_updates_dict, model_type='cnn', dataset='cifar10'):
    """使用 FedAvg 聚合来自多个参与者的模型更新"""
    print("\n>> Starting Local Model Aggregation...")
    if not participant_updates_dict:
        print("[ERROR] No participant updates provided for aggregation.")
        return None

    participant_models = []
    # 1. 反序列化所有有效的模型更新
    for p_id, model_str in participant_updates_dict.items():
        print(f">> Deserializing update from P:{p_id}...")
        model = deserialize_model(model_str, model_type=model_type, dataset=dataset)
        if model:
            # 聚合操作在 CPU 上进行，避免 GPU 显存问题
            participant_models.append(model.to(torch.device("cpu")))
        else:
            print(f"[WARN] Skipping invalid model update from participant {p_id}.")

    num_valid_updates = len(participant_models)
    if num_valid_updates == 0:
        print("[ERROR] No valid model updates found for aggregation.")
        return None

    print(f"\n>> Aggregating using FedAvg across {num_valid_updates} valid models...")

    # 2. 初始化聚合状态字典 (使用第一个有效模型的深拷贝)
    aggregated_state_dict = copy.deepcopy(participant_models[0].state_dict())

    # 3. 累加其他所有模型的参数权重
    #    (从索引 1 开始，因为索引 0 已作为基础)
    for model in participant_models[1:]:
        for name, param in model.state_dict().items():
            if name in aggregated_state_dict:
                aggregated_state_dict[name].add_(param) # 原地加法，效率更高
            else:
                # 如果模型结构不一致，可能会发生这种情况
                print(f"[WARN] Parameter '{name}' not found in base model state_dict during aggregation. Skipping.")

    # 4. 计算参数平均值
    for name in aggregated_state_dict:
        aggregated_state_dict[name].div_(num_valid_updates) # 原地除法

    # 5. 创建新的模型实例并加载聚合后的权重
    print(">> Creating new model instance for aggregated weights...")
    if model_type == 'mlp':
        aggregated_model = MLP()
    elif model_type == 'cnn':
        aggregated_model = CNN(num_classes=10) # 假设类别数
    else:
        print(f"[ERROR] Unsupported model type '{model_type}' for creating aggregated model.")
        return None

    aggregated_model.load_state_dict(aggregated_state_dict)
    print(">> Local Aggregation Finished Successfully!")
    return aggregated_model

# ========== (RESTORED) 链上 Keccak256 哈希计算函数 ==========
def calculate_keccak256_onchain(client, contract_abi, contract_address, text_string):
    """
    通过调用智能合约的 calculateStringHash 函数计算 Keccak256 哈希。
    返回 bytes 类型的哈希值，错误时返回 None。
    """
    fn_name = "calculateStringHash" # 确保与合约中的函数名一致
    args = [text_string]
    # print(f">> Calculating hash via contract call...") # 减少不必要的日志

    if not isinstance(text_string, str):
        print("[ERROR] Input to calculate_keccak256_onchain must be a string.")
        return None

    try:
        # 执行只读的合约调用
        result = client.call(contract_address, contract_abi, fn_name, args)
        # BCOS SDK 通常在结果外层包一个列表
        if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], bytes):
            hash_bytes = result[0]
            # print(f">> Hash from contract: {hash_bytes.hex()}") # 减少不必要的日志
            return hash_bytes
        else:
            print(f"[ERROR] Failed to calculate hash on chain. Unexpected result format: {result}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during calculateStringHash contract call: {e}")
        # traceback.print_exc() # 在调试时取消注释
        return None

# ========== 合约交互函数 ==========
def submit_update_to_contract(client, contract_abi, contract_address, model_update_str, round_num, participant_id):
    """提交参与者的模型更新到智能合约"""
    print(f"\n>> Submitting Update to Contract: P:{participant_id} R:{round_num}...")
    fn_name = "submitModelUpdate"
    args = [model_update_str, round_num, participant_id]
    try:
        receipt = client.sendRawTransaction(contract_address, contract_abi, fn_name, args)
        # 检查交易回执状态
        if receipt and receipt.get("status") == 0:
            tx_hash = receipt.get('transactionHash', 'N/A')
            print(f">> Submit Update OK. TxHash: {tx_hash}")
            return True, receipt
        else:
            # 打印失败信息
            status = receipt.get('status') if receipt else 'N/A'
            output = receipt.get('output', 'N/A') if receipt else 'N/A' # output 可能包含 revert 信息
            print(f"[ERROR] Submit Update Failed. Status: {status}, Output: {output[:100]}...") # 只显示部分 output
            return False, receipt
    except Exception as e:
        print(f"[ERROR] Exception during submitModelUpdate transaction: {e}")
        traceback.print_exc() # 打印完整错误栈
        return False, None

def get_updates_from_contract(client, contract_abi, contract_address, round_num):
    """从合约获取指定轮次的所有模型更新 (JSON 字符串并解析)"""
    print(f"\n>> Getting Participant Updates from Contract for Round {round_num}...")
    fn_name = "getParticipantUpdates"
    args = [round_num]
    try:
        result = client.call(contract_address, contract_abi, fn_name, args)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            json_string = result[0]
            print(f">> Received JSON string from contract (length: {len(json_string)}).")
            try:
                # 解析 JSON
                updates_list = json.loads(json_string)
                print(f">> JSON parsed successfully, found {len(updates_list)} updates.")
                # 转换为字典格式，方便后续处理
                updates_dict = {item['participantId']: item['modelUpdate'] for item in updates_list}
                return updates_dict
            except (json.JSONDecodeError, KeyError) as e:
                # 处理 JSON 解析错误或格式问题
                print(f"[ERROR] Failed to parse JSON updates string or key missing: {e}")
                print(f"  Received string sample: {json_string[:200]}...") # 打印部分原始字符串帮助调试
                return None
        else:
            print(f"[ERROR] Failed to get updates. Unexpected result format from contract call: {result}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during getParticipantUpdates contract call: {e}")
        traceback.print_exc()
        return None

def submit_hash_to_contract(client, contract_abi, contract_address, round_num, model_hash_bytes, participant_id):
    """尝试提交聚合模型的哈希到合约"""
    print(f"\n>> Attempting Submit Aggregation Hash: P:{participant_id} R:{round_num}...")
    fn_name = "submitAggregationHash"
    # 确保传入的是 bytes 类型
    if not isinstance(model_hash_bytes, bytes):
        print("[ERROR] Hash to submit must be of bytes type.")
        return False, None
    args = [round_num, model_hash_bytes, participant_id]
    try:
        receipt = client.sendRawTransaction(contract_address, contract_abi, fn_name, args)
        if receipt and receipt.get("status") == 0:
            tx_hash = receipt.get('transactionHash', 'N/A')
            print(f">> Submit Hash OK. TxHash: {tx_hash}")
            return True, receipt
        else:
            # 提交失败是正常情况（如果已有哈希）
            status = receipt.get('status') if receipt else 'N/A'
            output = receipt.get('output', 'N/A') if receipt else 'N/A' # output 可能包含 revert 原因
            print(f">> Submit Hash Failed (Expected if already submitted). Status:{status}, Output:{output[:100]}...")
            return False, receipt
    except Exception as e:
        print(f"[ERROR] Exception during submitAggregationHash transaction: {e}")
        traceback.print_exc()
        return False, None

def get_hash_from_contract(client, contract_abi, contract_address, round_num):
    """从合约获取指定轮次的官方聚合哈希"""
    # print(f">> Getting Hash from Contract for Round {round_num}...") # 可以减少日志输出
    fn_name = "getRoundModelHash"
    args = [round_num]
    try:
        result = client.call(contract_address, contract_abi, fn_name, args)
        if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], bytes):
            hash_bytes = result[0]
            # 检查是否是 Solidity bytes32 的零值
            if hash_bytes == b'\x00' * 32:
                # print(">> Hash has not been submitted for this round yet.") # 可以减少日志输出
                return None # 返回 None 表示哈希不存在
            else:
                # print(f">> Retrieved hash: {hash_bytes.hex()}") # 可以减少日志输出
                return hash_bytes # 返回获取到的哈希
        else:
            print(f"[ERROR] Get Hash failed for Round {round_num}. Unexpected result format: {result}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during getRoundModelHash call for Round {round_num}: {e}")
        return None

def get_latest_round_from_contract(client, contract_abi, contract_address):
    """从合约获取最新已提交聚合哈希的轮次编号"""
    fn_name = "getLatestAggregatedRound"
    args = []
    try:
        result = client.call(contract_address, contract_abi, fn_name, args)
        if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], int):
            return result[0] # 返回轮次编号
        else:
            print(f"[ERROR] Get Latest Round failed. Unexpected result: {result}")
            return -1 # 返回 -1 表示错误
    except Exception as e:
        # 对于频繁调用的函数，可以减少错误日志的冗余度
        # print(f"[WARN] Exception during getLatestAggregatedRound call: {e}")
        return -1

def get_participant_ids_from_contract(client, contract_abi, contract_address, round_num):
    """从合约获取指定轮次提交了更新的参与者 ID 列表"""
    fn_name = "getRoundParticipantIds" # 使用合约中的 getter 函数
    args = [round_num]
    try:
        result = client.call(contract_address, contract_abi, fn_name, args)
        # 期望返回 [[id1, id2, ...]]
        if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], (list, tuple)):
            participant_ids = list(result[0]) # 提取内部的列表
            return participant_ids
        else:
            print(f"[ERROR] Get Participant IDs failed R:{round_num}. Unexpected result: {result}")
            return [] # 返回空列表表示失败
    except Exception as e:
        print(f"[ERROR] Exception during getRoundParticipantIds call R:{round_num}: {e}")
        return []

# ========== 模型评估函数 ==========
def evaluate_model(model, test_loader, log_dir="fl_log", round_num=1, participant_id="local_eval"):
    """在测试数据集上评估模型性能"""
    if test_loader is None:
        print("[WARN] No test data provided for evaluation.")
        return 0.0

    print(f"\n>> Starting Evaluation: P:{participant_id} R:{round_num}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 设置为评估模式
    correct = 0
    total = 0

    try:
        with torch.no_grad(): # 评估时不需要计算梯度
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1) # 获取最大概率的类别
                total += target.size(0) # 累加样本总数
                correct += (predicted == target).sum().item() # 累加预测正确的样本数

        # 计算准确率
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # 记录评估结果到日志文件
        log_filename = os.path.join(log_dir, f"evaluation_log.txt")
        os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在
        with open(log_filename, 'a') as log_file:
            log_info = f'[Round {round_num}, Evaluator: {participant_id}] Test Accuracy: {accuracy:.2f}% ({correct}/{total})'
            print(f">>{log_info}") # 同时打印到控制台
            log_file.write(log_info + '\n')
        return accuracy # 返回准确率

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return 0.0 # 发生错误返回 0
    finally:
        # 评估结束后，恢复模型到训练模式，以防影响后续训练
        model.train()

# --- END OF FILE latest_utils.py ---