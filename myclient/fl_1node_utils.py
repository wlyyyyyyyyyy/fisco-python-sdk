# fl_utils.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F #  导入 F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io
import base64
import os #  导入 os 模块

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

# ========== 加载 CIFAR-10 训练数据集 (保持不变) ==========
def load_cifar10_data_partition(batch_size=64):
    print("\n>>Loading CIFAR-10 Training Data ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("\n>>CIFAR-10 Training Data Loaded.")
    return train_loader

# ========== 加载 CIFAR-10 测试数据集 (保持不变) ==========
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

# ========== 定义 CNN 模型 (保持不变) ==========
class CNN(nn.Module):
    def __init__(self, num_classes=10): #  CIFAR-10 默认 10 分类
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # CIFAR-10 images are 3-channel (RGB)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128) #  Adjusted input size for CIFAR-10 after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # num_classes for CIFAR-10

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ========== 模型序列化函数 (保持不变) ==========
def serialize_model(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    model_str = base64.b64encode(model_bytes).decode()
    return model_str

# ========== 模型反序列化函数 (保持不变) ==========
def deserialize_model(model_str, model_type='cnn'): #  添加 model_type 参数，默认为 'cnn'
    initial_model_placeholder = "Initial Model - Very Simple" #  定义初始模型占位符字符串
    if model_str == initial_model_placeholder: #  判断是否是初始模型字符串
        print(f"\n>>Initial Model Placeholder String Detected. Initializing new {model_type.upper()} model...") # 打印提示信息,  根据 model_type 提示模型类型
        if model_type == 'mlp': #  根据 model_type 初始化 MLP 或 CNN 模型
            model = MLP()
        elif model_type == 'cnn':
            model = CNN()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}") #  添加不支持的模型类型错误提示
        return model #  直接返回新的模型，不进行解码
    else: #  如果不是初始模型字符串，则进行 Base64 解码和模型加载 (之前的逻辑)
        if model_type == 'mlp': #  根据 model_type 反序列化为 MLP 或 CNN 模型
            model = MLP()
        elif model_type == 'cnn':
            model = CNN()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}") #  添加不支持的模型类型错误提示
        model_bytes = base64.b64decode(model_str.encode())
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        return model

# ========== 模型训练函数 (修改后，添加日志记录) ==========
def train_model(model, train_loader, epochs=1, log_dir="fl_single_log", round_num=1): #  添加 log_dir 和 round_num 参数
    print("\n>>Starting Local Model Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_filename = os.path.join(log_dir, f"train_log_round_{round_num}.txt") #  构建训练日志文件名
    os.makedirs(log_dir, exist_ok=True) #  确保日志文件夹存在

    model.train()
    with open(log_filename, 'a') as log_file: #  打开日志文件，追加模式
        for epoch in range(epochs): # 固定 epochs=1
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    log_info = f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}' #  构建日志信息
                    print(log_info) #  打印到终端
                    log_file.write(log_info + '\n') #  写入日志文件
        print("\n>>Local Model Training Finished!")
    return model

# ========== 模型评估函数 (修改后，添加日志记录) ==========
def evaluate_model(model, test_loader, log_dir="fl_single_log", round_num=1): #  添加 log_dir 和 round_num 参数
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
    log_filename = os.path.join(log_dir, f"evaluation_log_round_{round_num}.txt") #  构建评估日志文件名
    os.makedirs(log_dir, exist_ok=True) #  确保日志文件夹存在
    with open(log_filename, 'w') as log_file: #  打开日志文件，写入模式 (每次评估覆盖)
        log_info = f'\n>>Accuracy on Test Data: {accuracy:.2f}%' #  构建日志信息
        print(log_info) #  打印到终端
        log_file.write(log_info + '\n') #  写入日志文件
    print("\n>>Model Evaluation Finished!")
    return accuracy

# ========== 模型上传函数 (保持不变) ==========
def upload_model_update(client, contract_abi, contract_address, updated_model_str, role_name): #  保持 client, contract_abi, contract_address 参数
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
def download_global_model(client, contract_abi, contract_address, participant_address, role_name): #  保持 client, contract_abi, contract_address 参数
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