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

# ==========  操作日志记录函数 (保持不变) ==========
def log_operation(log_dir, round_num, role_name, operation_type, message):
    log_filename = os.path.join(log_dir, f"operations_log.txt")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_filename, 'a') as log_file:
        log_info = f'Round: {round_num}, Role: {role_name}, Operation: {operation_type}, Message: {message}'
        print(f"[OPERATION LOG] - {log_info}")
        log_file.write(log_info + '\n')

# ==========  MNIST 数据加载函数 (修改为支持多参与者数据划分) ==========
def load_mnist_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3): #  !!! 修改 total_participants 默认值为 3 !!!
    print(f"\n>>Loading MNIST Training Data for Participant: {participant_id} ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # -----  数据划分逻辑 (示例:  平均划分 -  支持多参与者) -----
    num_samples = len(full_train_dataset)
    partition_size = num_samples // total_participants
    start_index = (int(participant_id[-1]) - 1) * partition_size #  !!! participant_id 仍然假设为 "participant1", "participant2", "participant3" 格式 !!!
    end_index = start_index + partition_size
    if int(participant_id[-1]) == total_participants: #  !!!  处理最后一个参与者，可能分配剩余数据 !!!
        end_index = num_samples  #  最后一个参与者分配到数据集末尾

    indices = list(range(start_index, end_index))
    partitioned_dataset = Subset(full_train_dataset, indices)

    train_loader = DataLoader(partitioned_dataset, batch_size=batch_size, shuffle=True)
    print(f"\n>>MNIST Training Data Loaded for Participant: {participant_id} ({len(partitioned_dataset)} samples).")
    return train_loader


# ==========  CIFAR-10 数据加载函数 (修改为支持多参与者数据划分) ==========
def load_cifar10_data_partition_multi(batch_size=64, participant_id="participant1", total_participants=3): #  !!! 修改 total_participants 默认值为 3 !!!
    print(f"\n>>Loading CIFAR-10 Training Data for Participant: {participant_id} ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # -----  数据划分逻辑 (示例:  平均划分 -  支持多参与者) -----
    num_samples = len(full_train_dataset)
    partition_size = num_samples // total_participants
    start_index = (int(participant_id[-1]) - 1) * partition_size #  !!! participant_id 仍然假设为 "participant1", "participant2", "participant3" ... 格式 !!!
    end_index = start_index + partition_size
    if int(participant_id[-1]) == total_participants: #  !!!  处理最后一个参与者，可能分配剩余数据 !!!
        end_index = num_samples # 最后一个参与者分配到数据集末尾

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
    initial_model_placeholder = "Initial Model - Very Simple"
    if model_str == initial_model_placeholder:
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
def train_model(model, train_loader, epochs=1, log_dir="fl_multi_log", round_num=1, participant_id="participant1"): #  !!!  修改默认 log_dir !!!
    print(f"\n>>Starting Local Model Training for Participant: {participant_id} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_filename = os.path.join(log_dir, f"train_log_round_{round_num}_{participant_id}.txt") #  !!! 修改默认 log_dir !!!
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
def evaluate_model(model, test_loader, log_dir="fl_multi_log", round_num=1): #  !!!  修改默认 log_dir !!!
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
    log_filename = os.path.join(log_dir, f"evaluation_log_round_{round_num}.txt") #  !!!  修改默认 log_dir !!!
    os.makedirs(log_dir, exist_ok=True)
    with open(log_filename, 'w') as log_file:
        log_info = f'\n>>Accuracy on Test Data (Global Model): {accuracy:.2f}%'
        print(log_info)
        log_file.write(log_info + '\n')
    print("\n>>Global Model Evaluation Finished!")
    return accuracy


# ==========  模型上传函数 (保持不变) ==========
def upload_model_update(client, contract_abi, contract_address, updated_model_str, role_name, round_num):
    print(f"\n>>Uploading Model Update (from Role: {role_name}, Round: {round_num})...")
    to_address = contract_address
    fn_name = "updateModel"
    args = [updated_model_str, round_num, role_name]
    receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args)
    if receipt is not None and receipt["status"] == 0:
        print(f"Upload Success (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return True
    else:
        print(f"Upload Failed (from Role: {role_name}, Round: {round_num}), receipt: {receipt}")
        return False


# ==========  模型下载函数 (保持不变) ==========
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