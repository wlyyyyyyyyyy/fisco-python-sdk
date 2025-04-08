import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import datetime

# ===== 日志记录函数 =====
def log_operation(log_dir, round_num, role, operation, message):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(log_dir, f"{role}_round_{round_num}.log")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] - {operation}: {message}\n")

# ===== MNIST 数据加载函数 (多参与者) =====
def load_mnist_data_partition_multi(participant_id, total_participants, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    num_samples = len(train_dataset)
    samples_per_participant = num_samples // total_participants
    remaining_samples = num_samples % total_participants
    indices = list(range(num_samples))
    random.shuffle(indices)
    start_index = participant_id * samples_per_participant + min(participant_id, remaining_samples)
    end_index = (participant_id + 1) * samples_per_participant + min(participant_id + 1, remaining_samples)
    participant_indices = indices[start_index:end_index]
    participant_data = torch.utils.data.Subset(train_dataset, participant_indices)
    train_loader = DataLoader(participant_data, batch_size=batch_size, shuffle=True)
    return train_loader

# ===== MNIST 测试数据加载函数 =====
def load_mnist_test_data(batch_size=1000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# ===== CIFAR-10 数据加载函数 (多参与者) =====
def load_cifar10_data_partition_multi(participant_id, total_participants, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    num_samples = len(train_dataset)
    samples_per_participant = num_samples // total_participants
    remaining_samples = num_samples % total_participants
    indices = list(range(num_samples))
    random.shuffle(indices)
    start_index = participant_id * samples_per_participant + min(participant_id, remaining_samples)
    end_index = (participant_id + 1) * samples_per_participant + min(participant_id + 1, remaining_samples)
    participant_indices = indices[start_index:end_index]
    participant_data = torch.utils.data.Subset(train_dataset, participant_indices)
    train_loader = DataLoader(participant_data, batch_size=batch_size, shuffle=True)
    return train_loader

# ===== CIFAR-10 测试数据加载函数 =====
def load_cifar10_test_data(batch_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# ===== MLP 模型定义 (用于 MNIST) =====
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# ===== CNN 模型定义 (用于 CIFAR-10) =====
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== 模型序列化函数 =====
def serialize_model(model):
    return json.dumps({k: v.tolist() for k, v in model.state_dict().items()})

# ===== 模型反序列化函数 =====
def deserialize_model(model_str, model_type='mlp'):
    if model_type == 'mlp':
        model = MLP()
    elif model_type == 'cnn':
        model = CNN()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    state_dict = {k: torch.tensor(v) for k, v in json.loads(model_str).items()}
    model.load_state_dict(state_dict)
    return model

# ===== 模型训练函数 =====
def train_model(model, train_loader, epochs, log_dir, round_num, participant_id):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                log_operation(log_dir, round_num, participant_id, f"Epoch {epoch+1}", f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
    return model

# ===== 模型评估函数 =====
def evaluate_model(model, test_loader, log_dir, round_num, participant_id="server"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    log_operation(log_dir, round_num, participant_id, "Evaluation", f"Accuracy on test data: {accuracy:.2f}%")
    print(f"\n>>[{participant_id}] Round {round_num} - Test Accuracy: {accuracy:.2f}%")

# ===== 模型更新上传函数 =====
def upload_model_update(client, contract_abi, contract_address, model_update_str, participant_id, round_num):
    print(f"\n>>[{participant_id}] Round {round_num} - Uploading model update.")
    try:
        tx_hash = client.sendRawTransaction(
            contract_address,
            contract_abi,
            "uploadModelUpdate",
            [round_num, model_update_str]
        )
        print(f"\n>>[{participant_id}] Round {round_num} - Model update transaction hash: {tx_hash}")
        receipt = client.getTransactionReceipt(tx_hash)
        print(f"\n>>[{participant_id}] Round {round_num} - Model update receipt: {receipt}")
        if receipt and receipt['status'] == 0:
            print(f"\n>>[{participant_id}] Round {round_num} - Model update submitted successfully.")
            return True
        else:
            print(f"\n>>[{participant_id}] Round {round_num} - Failed to submit model update.")
            return False
    except Exception as e:
        print(f"\n>>[{participant_id}] Round {round_num} - Error uploading model update: {e}")
        return False

# ===== 提交本地全局模型哈希函数 =====
def submit_local_global_model_hash(client, contract_abi, contract_address, model_hash, participant_id, round_num):
    print(f"\n>>[{participant_id}] Round {round_num} - Submitting model hash.")
    try:
        tx_hash = client.sendRawTransaction(
            contract_address,
            contract_abi,
            "submitLocalGlobalModelHash",
            [round_num, model_hash]
        )
        print(f"\n>>[{participant_id}] Round {round_num} - Model hash transaction hash: {tx_hash}")
        receipt = client.getTransactionReceipt(tx_hash)
        print(f"\n>>[{participant_id}] Round {round_num} - Model hash receipt: {receipt}")
        if receipt and receipt['status'] == 0:
            print(f"\n>>[{participant_id}] Round {round_num} - Model hash submitted successfully.")
            return True
        else:
            print(f"\n>>[{participant_id}] Round {round_num} - Failed to submit model hash.")
            return False
    except Exception as e:
        print(f"\n>>[{participant_id}] Round {round_num} - Error submitting model hash: {e}")
        return False
    
# ===== 去中心化模型聚合函数 =====
def aggregate_global_model_decentralized(current_model, participant_updates):
    """
    Decentralized aggregation: Average the weights of the current model with updates from other participants.
    """
    total_participants = len(participant_updates) + 1 # Plus 1 for the current participant (whose model was just trained)
    if total_participants == 0:
        return current_model

    with torch.no_grad():
        # Aggregate with the current participant's trained model
        aggregated_state_dict = current_model.state_dict()
        for name, param in aggregated_state_dict.items():
            param.data.zero_() # Initialize with zeros

        # Add the current participant's trained model weights
        for name, param in current_model.named_parameters():
            if name in aggregated_state_dict:
                aggregated_state_dict[name].data += param.data

        # Add the weights from other participants' updates
        for update_model in participant_updates.values():
            for name, param in update_model.named_parameters():
                if name in aggregated_state_dict:
                    aggregated_state_dict[name].data += param.data

        # Average the weights
        for name, param in aggregated_state_dict.items():
            param.data /= total_participants

        current_model.load_state_dict(aggregated_state_dict)

    return current_model
