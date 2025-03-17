# fl_multi_utils.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import io
import base64
import os
import json
import hashlib #  !!! 导入 hashlib 库 !!!


# ==========  操作日志记录函数 (保持不变) ==========
def log_operation(log_dir, round_num, role_name, operation_type, message):
    log_filename = os.path.join(log_dir, f"operations_log.txt")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_filename, 'a') as log_file:
        log_info = f'Round: {round_num}, Role: {role_name}, Operation: {operation_type}, Message: {message}'
        print(f"[OPERATION LOG] - {log_info}")
        log_file.write(log_info + '\n')

# ==========  MNIST 数据加载函数 (保持不变) ==========
def load_mnist_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3):
    print(f"\n>>Loading MNIST Training Data for Participant: {participant_id} ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    num_samples = len(full_train_dataset)
    partition_size = num_samples // total_participants
    start_index = (int(participant_id[-1]) - 1) * partition_size
    end_index = start_index + partition_size
    if int(participant_id[-1]) == total_participants:
        end_index = num_samples

    indices = list(range(start_index, end_index))
    partitioned_dataset = Subset(full_train_dataset, indices)

    train_loader = DataLoader(partitioned_dataset, batch_size=batch_size, shuffle=True)
    print(f"\n>>MNIST Training Data Loaded for Participant: {participant_id} ({len(partitioned_dataset)} samples).")
    return train_loader

# ==========  CIFAR-10 数据加载函数 (保持不变) ==========
def load_cifar10_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3):
    print(f"\n>>Loading CIFAR-10 Training Data for Participant: {participant_id} ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    num_samples = len(full_train_dataset)
    partition_size = num_samples // total_participants
    start_index = (int(participant_id[-1]) - 1) * partition_size
    end_index = start_index + partition_size
    if int(participant_id[-1]) == total_participants:
        end_index = num_samples

    indices = list(range(start_index, end_index))
    partitioned_dataset = Subset(full_train_dataset, indices)

    train_loader = DataLoader(partitioned_dataset, batch_size=batch_size, shuffle=True)
    print(f"\n>>CIFAR-10 Training Data Loaded for Participant: {participant_id} ({len(partitioned_dataset)} samples).")
    return train_loader

# ==========  测试数据集加载函数 (MNIST 和 CIFAR-10，保持不变) ==========
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

def load_cifar10_test_data(batch_size=64):
    print("\n>>Loading CIFAR-10 Test Data ...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("\n>>CIFAR-10 Test Data Loaded.")
    return test_loader

# ==========  模型定义 (MLP 和 CNN，保持不变) ==========
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ==========  模型序列化和反序列化 (保持不变) ==========
def serialize_model(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    model_str = base64.b64encode(model_bytes).decode()
    return model_str

def deserialize_model(model_str, model_type='cnn'):
    initial_model_placeholder = "Initial Model - Enhanced"
    print(f"\n>>[DEBUG - deserialize_model] model_str (repr): {repr(model_str)}") # DEBUG - Print input model_str
    print(f"\n>>[DEBUG - deserialize_model] initial_model_placeholder (repr): {repr(initial_model_placeholder)}") # DEBUG - Print placeholder string
    model_str_stripped = model_str.strip() # 去除 model_str 前后空白
    initial_model_placeholder_stripped = initial_model_placeholder.strip() # 去除 placeholder 前后空白
    if model_str_stripped == initial_model_placeholder_stripped: # 使用去除空白后的字符串进行比较
        print(f"\n>>Initial Model Placeholder String Detected. Initializing new {model_type.upper()} model...")
        if model_type == 'mlp':
            model = MLP()
        elif model_type == 'cnn':
            model = CNN()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return model
    else:
        if model_type == 'mlp':
            model = MLP()
        elif model_type == 'cnn':
            model = CNN()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model_bytes = base64.b64decode(model_str.encode())
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        return model

# ==========  模型训练函数 (保持不变) ==========
def train_model(model, train_loader, epochs=1, log_dir="fl_decentralized_log", round_num=1, participant_id="participant1"): #  !!!  日志文件夹修改为 fl_decentralized_log !!!
    print(f"\n>>Starting Local Model Training for Participant: {participant_id} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_filename = os.path.join(log_dir, f"train_log_round_{round_num}_{participant_id}.txt") #  !!! 日志文件夹修改为 fl_decentralized_log !!!
    os.makedirs(log_dir, exist_ok=True)

    model.train()
    with open(log_filename, 'a') as log_file:
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    log_info = f'Participant: {participant_id}, Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}'
                    print(log_info)
                    log_file.write(log_info + '\n')
        print(f"\n>>Local Model Training Finished for Participant: {participant_id}!")
    return model

# ==========  模型评估函数 (保持不变) ==========
def evaluate_model(model, test_loader, log_dir="fl_decentralized_log", round_num=1, role_name="participant"): #  !!!  日志文件夹修改为 fl_decentralized_log,  新增 role_name 参数， 默认 participant !!!
    print("\n>>Starting Global Model Evaluation on Test Data...")
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
    log_filename = os.path.join(log_dir, f"evaluation_log_round_{round_num}_{role_name}.txt") #  !!!  日志文件夹修改为 fl_decentralized_log,  文件名包含 role_name !!!
    os.makedirs(log_dir, exist_ok=True)
    with open(log_filename, 'w') as log_file:
        log_info = f'\n>>Accuracy on Test Data (Local Global Model - {role_name}): {accuracy:.2f}%' #  !!!  日志信息包含 role_name,  修改为 Local Global Model !!!
        print(log_info)
        log_file.write(log_info + '\n')
    print("\n>>Global Model Evaluation Finished!") #  !!! 打印信息保持 "Global Model Evaluation Finished!"，  但实际是本地全局模型评估 !!!
    return accuracy

# ==========  模型上传函数 (保持不变) ==========
def upload_model_update(client, contract_abi, contract_address, updated_model_str, role_name, round_num):
    print(f"\n>>Uploading Model Update (from Role: {role_name}, Round: {round_num})...")
    to_address = contract_address
    fn_name = "updateModel"
    args = [updated_model_str, role_name]
    receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args)
    if receipt is not None and receipt["status"] == 0:
        print(f"Upload Success (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return True
    else:
        print(f"Upload Failed (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return False

# ==========  全局模型哈希上传函数 (保持不变) ==========
def upload_global_model_hash(client, contract_abi, contract_address, model_hash, role_name, round_num):
    print(f"\n>>Uploading Global Model Hash (from Role: {role_name}, Round: {round_num})...")
    to_address = contract_address
    fn_name = "submitGlobalModelHash"
    args = [model_hash.hex(), role_name]
    receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args)
    if receipt is not None and receipt["status"] == 0:
        print(f"Upload Global Model Hash Success (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return True
    else:
        print(f"Upload Global Model Hash Failed (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return False


# ==========  模型聚合函数 (简单的平均聚合) -  保持不变 ==========
def aggregate_global_model(global_model, participant_updates, model_type='cnn'):
    print("\n>>Starting Global Model Aggregation (Decentralized Version)...") #  !!! 修改打印信息， 强调 Decentralized Version !!!
    aggregated_state_dict = {}
    participant_models = []

    initial_model_state_dict = global_model.state_dict() #  !!!  获取初始模型的状态字典作为基准 !!!

    #  使用初始模型作为基准， 如果没有模型更新， 则直接返回初始模型， 避免聚合出错
    participant_models.append(global_model) #  !!!  将初始模型加入到 participant_models 列表中 !!!


    if not participant_updates or len(participant_updates) == 0: #  !!!  如果没有模型更新， 则直接返回初始模型 !!!
        print("\n>>No participant updates available for aggregation. Returning initial model.") #  !!!  打印提示信息 !!!
        return global_model #  !!!  直接返回初始模型 !!!

    for update in participant_updates:
        participant_id = update['participantId']
        model_update_str = update['modelUpdate']
        print(f"\n>>Deserializing model update from participant: {participant_id}...")
        participant_model = deserialize_model(model_update_str, model_type=model_type)
        participant_models.append(participant_model)

    with torch.no_grad():
        for name, param in initial_model_state_dict.items(): #  !!!  使用初始模型的状态字典初始化 aggregated_state_dict !!!
            aggregated_state_dict[name] = torch.zeros_like(param) #  !!!  使用 zeros_like 初始化 !!!

        for participant_model in participant_models: #  !!!  包含初始模型和所有参与者模型 !!!
            for name, param in participant_model.state_dict().items():
                if name in aggregated_state_dict: #  !!!  添加判断， 避免 KeyError !!!
                    aggregated_state_dict[name] += param

        num_participants = len(participant_models) #  !!!  聚合时， 除以 参与者模型数量 (包含初始模型) !!!
        for name, param in aggregated_state_dict.items():
             if name in aggregated_state_dict: #  !!!  再次添加判断， 确保安全 !!!
                aggregated_state_dict[name] /= num_participants

        global_model.load_state_dict(aggregated_state_dict)

    print("\n>>Global Model Aggregation Finished! (Decentralized Version)") #  !!! 修改打印信息， 强调 Decentralized Version !!!
    return global_model


# ==========  计算模型哈希值的函数 (保持不变) ==========
def calculate_model_hash(model):
    model_str = serialize_model(model)
    model_bytes = model_str.encode('utf-8')
    model_hash = hashlib.sha256(model_bytes).digest()
    return model_hash