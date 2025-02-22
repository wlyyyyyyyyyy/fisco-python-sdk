pragma solidity ^0.7.6;

contract FederatedLearning {
    string private globalModel;
    address public centralNodeAddress;
    mapping(address => bool) public registeredParticipants;
    address public participantList;
    mapping(address => string) public participantModelUpdates; // 存储每个参与者提交的模型更新

    constructor() {
        globalModel = "Initial Global Model - Federated Averaging"; // Updated initial model description
        centralNodeAddress = msg.sender; // Deployer is central node
    }

    // 参与者注册 (保持不变)
    function registerParticipant() public {
        require(!registeredParticipants[msg.sender], "Participant already registered");
        registeredParticipants[msg.sender] = true;
        participantList.push(msg.sender);
    }

    // 中央节点检查参与者是否已注册 (保持不变)
    function isParticipantRegistered(address participantAddress) public view returns (bool) {
        return registeredParticipants[participantAddress];
    }

    // 获取参与者列表 (仅中央节点可调用) (保持不变)
    function getParticipantList() public view returns (address memory) {
        require(msg.sender == centralNodeAddress, "Only central node can get participant list");
        return participantList;
    }

    // 获取全局模型 (注册参与者可调用) (保持不变)
    function requestGlobalModel(address participantAddress) public view returns (string memory) {
        require(registeredParticipants[participantAddress], "Participant not registered");
        return globalModel;
    }

    // 训练节点提交模型更新 (修改: 存储模型更新)
    function submitModelUpdate(string memory modelUpdate) public {
        require(registeredParticipants[msg.sender], "Participant not registered");
        participantModelUpdates[msg.sender] = modelUpdate; // Store model update from participant
    }

    // 中央节点触发模型聚合 (修改:  Placeholder for future on-chain aggregation logic)
    function aggregateModel() public {
        require(msg.sender == centralNodeAddress, "Only central node can aggregate model");
        // In a real FedAvg scenario, the aggregation logic (like averaging model weights)
        // would be implemented here. However, with string models, we cannot perform
        // numerical averaging in Solidity effectively.

        // For this simplified example, the on-chain aggregateModel function
        // can act as a trigger or placeholder. The actual FedAvg calculation
        // will be performed off-chain in the client script.

        // For now, we can simply reset the participantModelUpdates mapping
        // to prepare for the next round of updates.
        // We can also emit an event to signal that aggregation is triggered.

        // Reset participant model updates for the next round
        for (uint i = 0; i < participantList.length; i++) {
            delete participantModelUpdates[participantList[i]];
        }

        // In a more advanced version, you might emit an event here to signal
        // that aggregation has been triggered.
        // emit LogAggregationTriggered("Model aggregation triggered by central node.");
    }

    // 获取全局模型 (任何人可调用, 用于评估或其他用途) (保持不变)
    function getModel() public view returns (string memory) {
        return globalModel;
    }

    // 中央节点更新全局模型 (Re-introduced: for uploading the averaged model)
    function updateGlobalModel(string memory updatedModel) public {
        require(msg.sender == centralNodeAddress, "Only central node can update global model");
        globalModel = updatedModel; // Central node updates the global model with the averaged model
    }

    // 中央节点获取所有参与者的模型更新 (New function)
    function getParticipantModelUpdates() public view returns (string memory) {
        require(msg.sender == centralNodeAddress, "Only central node can get participant updates");
        uint256 participantCount = participantList.length;
        string memory updates = new string(participantCount);
        for (uint256 i = 0; i < participantCount; i++) {
            updates[i] = participantModelUpdates[participantList[i]];
        }
        return updates;
    }
}