pragma solidity ^0.7.6;

contract EnhancedFederatedLearning {

    mapping(uint => string) public globalModelUpdates;
    mapping(uint => mapping(string => string)) public participantModelUpdates; // Nested mapping to store updates per round and participant
    string public initialModel = "Initial Model - Enhanced";
    mapping(uint => string[]) public roundParticipantIds; //  **新增：存储每个 round 的 participantId 数组**


    constructor() public {
        globalModelUpdates[0] = initialModel; // Initialize round 0 with initial model, using round 0 explicitly
    }

    function updateModel(string memory modelUpdate, uint roundNumber, string memory roleName) public {
        if (keccak256(abi.encodePacked(roleName)) == keccak256(abi.encodePacked("server"))) {
            globalModelUpdates[roundNumber] = modelUpdate;
        } else {
            participantModelUpdates[roundNumber][roleName] = modelUpdate;
            
            bool alreadyAdded = false;
            string[] storage ids = roundParticipantIds[roundNumber];
            for (uint256 i = 0; i < ids.length; i++) {
                if (keccak256(abi.encodePacked(ids[i])) == keccak256(abi.encodePacked(roleName))) {
                    alreadyAdded = true;
                    break;
                }
            }
            if (!alreadyAdded) {
                roundParticipantIds[roundNumber].push(roleName);
            }
        }
    }

    function getModel() public view returns (string memory) {
        uint currentRound = getCurrentRound();
        string memory currentGlobalModel = globalModelUpdates[currentRound];
        if (bytes(currentGlobalModel).length > 0) {
            return currentGlobalModel;
        } else {
            return initialModel;
        }
    }


    function getParticipantUpdates(uint roundNumber) public view returns (string memory) {
        mapping(string => string) storage updates = participantModelUpdates[roundNumber];
        string memory jsonOutput = "["; // Start JSON array
        bool firstPair = true;

        string[] storage participantIds = roundParticipantIds[roundNumber]; // **从 roundParticipantIds 获取 participantId 数组**

        for (uint256 i = 0; i < participantIds.length; i++) { // 使用索引 for 循环遍历 participantIds 数组
            string memory participantId = participantIds[i];
            string memory modelUpdate = updates[participantId]; // 直接通过 participantId 获取 modelUpdate

            if (bytes(modelUpdate).length > 0) { // 仅当 modelUpdate 存在时才添加到 JSON
                if (!firstPair) {
                    jsonOutput = string(abi.encodePacked(jsonOutput, ",")); // 添加逗号分隔符
                }
                // Properly encode participantId and modelUpdate as JSON strings
                string memory participantIdJson = string(abi.encodePacked('"', participantId, '"'));
                string memory modelUpdateJson = string(abi.encodePacked('"', updates[participantId], '"'));

                jsonOutput = string(abi.encodePacked(jsonOutput, '{"participantId":', participantIdJson, ',"modelUpdate":', modelUpdateJson, '}'));
                firstPair = false;
            }
        }

        jsonOutput = string(abi.encodePacked(jsonOutput, "]")); // End JSON array
        return jsonOutput;
    }


    function getCurrentRound() public view returns (uint) {
        uint round = 0;
        while (bytes(globalModelUpdates[round + 1]).length!= 0) {
            round++;
        }
        return round;
    }
}