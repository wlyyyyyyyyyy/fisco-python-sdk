// --- START OF FILE Latest.sol ---
pragma solidity ^0.7.6; // 你的编译器版本

// +++ 添加 ABIEncoderV2 启用指令 +++
pragma experimental ABIEncoderV2; // <--- 添加这一行

contract Latest { // 合约名与文件名匹配

    // === 状态变量 ===
    mapping(uint => mapping(string => string)) public participantModelUpdates;
    string public initialModel = "Initial Model - Enhanced";
    mapping(uint => string[]) public roundParticipantIds;
    mapping(uint => bytes32) public roundAggregatedModelHash;
    mapping(uint => string) public roundAggregatorId;

    // === 事件 ===
    event ParticipantUpdateSubmitted(uint indexed roundNumber, string participantId, uint updateLength);
    event AggregationSubmitted(uint indexed roundNumber, string indexed aggregatorId, bytes32 modelHash);

    constructor() {
        roundAggregatedModelHash[0] = keccak256(abi.encodePacked(initialModel));
    }

    function submitModelUpdate(string memory modelUpdate, uint roundNumber, string memory participantId) public {
        require(roundNumber > 0, "Round number must be > 0");
        require(roundAggregatedModelHash[roundNumber] == bytes32(0), "Aggregation already submitted");
        participantModelUpdates[roundNumber][participantId] = modelUpdate;
        // Add participantId to roundParticipantIds if not exists...
        bool found = false;
        for (uint i = 0; i < roundParticipantIds[roundNumber].length; i++) {
            if (keccak256(abi.encodePacked(roundParticipantIds[roundNumber][i])) == keccak256(abi.encodePacked(participantId))) {
                found = true; break;
            }
        }
        if (!found) { roundParticipantIds[roundNumber].push(participantId); }
        emit ParticipantUpdateSubmitted(roundNumber, participantId, bytes(modelUpdate).length);
    }

    function submitAggregationHash(uint roundNumber, bytes32 modelHash, string memory submitterId) public {
        require(roundNumber > 0, "Round number must be > 0");
        require(roundAggregatedModelHash[roundNumber] == bytes32(0), "Aggregation already submitted");
        roundAggregatedModelHash[roundNumber] = modelHash;
        roundAggregatorId[roundNumber] = submitterId;
        emit AggregationSubmitted(roundNumber, submitterId, modelHash);
    }

    function getParticipantUpdates(uint roundNumber) public view returns (string memory) {
        // ... (实现不变) ...
        mapping(string => string) storage updates = participantModelUpdates[roundNumber];
        string memory jsonOutput = "["; bool first = true; string[] storage pIds = roundParticipantIds[roundNumber];
        for (uint i = 0; i < pIds.length; i++) {
            string memory pId = pIds[i]; string memory update = updates[pId];
            if (bytes(update).length > 0) {
                if (!first) { jsonOutput = string(abi.encodePacked(jsonOutput, ",")); }
                string memory pIdJson = string(abi.encodePacked('"', pId, '"'));
                string memory updateJson = string(abi.encodePacked('"', update, '"'));
                jsonOutput = string(abi.encodePacked(jsonOutput, '{"participantId":', pIdJson, ',"modelUpdate":', updateJson, '}'));
                first = false;
            }
        }
        jsonOutput = string(abi.encodePacked(jsonOutput, "]")); return jsonOutput;
    }

    function getLatestAggregatedRound() public view returns (uint) {
        // ... (实现不变) ...
        uint latest = 0; uint current = 1;
        while(roundAggregatedModelHash[current] != bytes32(0)) { latest = current; current++; }
        return latest;
    }

    function getRoundModelHash(uint roundNumber) public view returns (bytes32) {
        return roundAggregatedModelHash[roundNumber];
    }

    // 这个函数现在可以正常编译了
    function getRoundParticipantIds(uint roundNumber) public view returns (string[] memory) {
        return roundParticipantIds[roundNumber];
    }

    function calculateStringHash(string memory inputString) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(inputString));
    }
}
// --- END OF FILE Latest.sol ---