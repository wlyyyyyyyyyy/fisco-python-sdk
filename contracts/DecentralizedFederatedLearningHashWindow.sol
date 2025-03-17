pragma solidity ^0.7.6;

import "openzeppelin-solidity/contracts/utils/Strings.sol";

contract DecentralizedFederatedLearningHashWindow {

    using Strings for string;

    uint public currentRound = 1;          // 当前轮数
    uint public modelUpdateWindowSize = 5; // 模型更新窗口大小，例如最多保存最近 5 轮的更新

    // 使用窗口索引来存储模型更新和哈希值
    mapping(uint => mapping(string => string)) public participantModelUpdatesWindow; //  使用窗口索引
    mapping(uint => mapping(string => bytes32)) public participantGlobalModelHashesWindow; // 使用窗口索引
    string[] public participantIds;
    uint public minParticipantsForAggregation;


    event ModelUpdateUploaded(uint round, string participantId, string modelUpdateHash, uint windowIndex);
    event GlobalModelHashUploaded(uint round, string participantId, bytes32 modelHash, uint windowIndex);
    event RoundAggregationStarted(uint round, uint windowIndex);
    event RoundAggregationFinished(uint round, bytes32 consensusModelHash, uint windowIndex);
    event NewRoundStarted(uint round, uint nextRoundWindowIndex);
    event ConsensusFailed(uint round, uint windowIndex);
    event ModelUpdateWindowSizeSet(uint newWindowSize);


    constructor(string[] memory _participantIds, uint _windowSize) public {
        require(_windowSize > 0, "Window size must be greater than 0.");
        participantIds = _participantIds;
        modelUpdateWindowSize = _windowSize;
        minParticipantsForAggregation = _participantIds.length;
        // 初始化第一轮的参与者列表
        for (uint i = 0; i < _participantIds.length; i++) {
            roundParticipantIds[1].push(_participantIds[i]);
        }
    }


    //  为了方便，保留了 roundParticipantIds 映射，但实际上简化版本中，参与者是固定的，可以考虑移除 roundParticipantIds 映射
    mapping(uint => string[]) public roundParticipantIds; // 存储参与到每一轮的参与者 ID 列表 (简化版本中，所有轮次参与者相同，这里实际上可以简化)


    modifier onlyParticipant(string memory participantId) {
        bool isParticipant = false;
        for (uint256 i = 0; i < participantIds.length; i++) {
            if (Strings.equal(participantIds[i], participantId)) {
                isParticipant = true;
                break;
            }
        }
        require(isParticipant, "Only registered participants can call this function.");
        _;
    }

    modifier onlyRegisteredParticipantForRound(string memory participantId, uint round) {
        uint windowIndex = getWindowIndex(round); //  计算窗口索引
        bool isRegistered = false;
        string[] storage ids = roundParticipantIds[round]; //  仍然使用 roundParticipantIds[round]
        for (uint256 i = 0; i < ids.length; i++) {
            if (Strings.equal(ids[i], participantId)) {
                isRegistered = true;
                break;
            }
        }
        require(isRegistered, "Participant not registered for this round.");
        _;
    }


    function updateModel(string memory modelUpdate, string memory participantId) public onlyParticipant(participantId) onlyRegisteredParticipantForRound(participantId, currentRound) {
        uint windowIndex = getWindowIndex(currentRound); //  计算当前轮次的窗口索引
        participantModelUpdatesWindow[windowIndex][participantId] = modelUpdate; //  使用窗口索引存储
        emit ModelUpdateUploaded(currentRound, participantId, keccak256(abi.encodePacked(modelUpdate)), windowIndex);
    }


    function submitGlobalModelHash(bytes32 modelHash, string memory participantId) public onlyParticipant(participantId) onlyRegisteredParticipantForRound(participantId, currentRound) {
        uint windowIndex = getWindowIndex(currentRound); //  计算当前轮次的窗口索引
        participantGlobalModelHashesWindow[windowIndex][participantId] = modelHash; //  使用窗口索引存储
        emit GlobalModelHashUploaded(currentRound, participantId, modelHash, windowIndex);

        checkAggregationAndStartNextRound();
    }


    function getParticipantUpdatesForAggregation(uint roundNumber) public view returns (string memory) {
        uint windowIndex = getWindowIndex(roundNumber); //  计算指定轮次的窗口索引
        require(roundParticipantIds[roundNumber].length >= minParticipantsForAggregation, "Not enough participants registered for aggregation yet.");
        mapping(string => string) storage updates = participantModelUpdatesWindow[windowIndex]; //  使用窗口索引获取
        string memory jsonOutput = "[";
        bool firstPair = true;
        string[] storage participantIdsInRound = roundParticipantIds[roundNumber];

        for (uint256 i = 0; i < participantIdsInRound.length; i++) {
            string memory participantId = participantIdsInRound[i];
            string memory modelUpdate = updates[participantId];
            if (bytes(modelUpdate).length > 0) {
                if (!firstPair) {
                    jsonOutput = string(abi.encodePacked(jsonOutput, ","));
                }
                string memory participantIdJson = string(abi.encodePacked('"', participantId, '"'));
                string memory modelUpdateJson = string(abi.encodePacked('"', updates[participantId], '"'));
                jsonOutput = string(abi.encodePacked(jsonOutput, '{"participantId":', participantIdJson, ',"modelUpdate":', modelUpdateJson, '}'));
                firstPair = false;
            }
        }
        jsonOutput = string(abi.encodePacked(jsonOutput, "]"));
        return jsonOutput;
    }


    function checkAggregationAndStartNextRound() private {
        uint roundNumber = currentRound;
        uint windowIndex = getWindowIndex(roundNumber); //  计算当前轮次的窗口索引
        string[] storage participantIdsInRound = roundParticipantIds[roundNumber];
        if (participantIdsInRound.length < minParticipantsForAggregation) {
            return;
        }

        uint submittedHashCount = 0;
        bytes32 firstHash = bytes32(0);
        bool allHashesMatch = true;
        bool firstHashFound = false;

        for (uint256 i = 0; i < participantIdsInRound.length; i++) {
            string memory participantId = participantIdsInRound[i];
            bytes32 modelHash = participantGlobalModelHashesWindow[windowIndex][participantId]; //  使用窗口索引获取
            if (modelHash != bytes32(0)) {
                submittedHashCount++;
                if (!firstHashFound) {
                    firstHash = modelHash;
                    firstHashFound = true;
                } else if (firstHash != modelHash) {
                    allHashesMatch = false;
                }
            } else {
                allHashesMatch = false;
            }
        }


        if (submittedHashCount < minParticipantsForAggregation) {
            emit RoundAggregationStarted(roundNumber, windowIndex); //  传递窗口索引
            return;
        }


        if (allHashesMatch && submittedHashCount == minParticipantsForAggregation) {
            emit RoundAggregationFinished(roundNumber, firstHash, windowIndex); //  传递窗口索引
            currentRound++;
            if (currentRound <= 5) {
                // 初始化下一轮的参与者列表 (与第一轮相同)
                for (uint i = 0; i < participantIds.length; i++) {
                    roundParticipantIds[currentRound].push(participantIds[i]);
                }
                uint nextRoundWindowIndex = getWindowIndex(currentRound); //  计算下一轮的窗口索引
                emit NewRoundStarted(currentRound, nextRoundWindowIndex); // 传递下一轮的窗口索引

                //  !!!  数据清理:  清理 "最旧" 窗口索引对应的数据  !!!
                uint windowIndexToDelete = nextRoundWindowIndex; //  下一轮的窗口索引，就是本轮将要被覆盖的 "最旧" 索引
                delete participantModelUpdatesWindow[windowIndexToDelete]; //  清理模型更新数据
                delete participantGlobalModelHashesWindow[windowIndexToDelete]; //  清理全局模型哈希数据


            } else {
                //  超过轮数限制，流程结束
            }


        } else {
            emit RoundAggregationStarted(roundNumber, windowIndex); //  传递窗口索引
            emit ConsensusFailed(roundNumber, windowIndex); //  传递窗口索引
        }
    }


    //  !!!  新增函数:  根据轮数计算窗口索引  !!!
    function getWindowIndex(uint round) private view returns (uint) {
        //  窗口索引计算公式:  (round - 1) % windowSize + 1
        //  确保窗口索引在 1 到 windowSize 之间循环
        return (round - 1) % modelUpdateWindowSize + 1;
    }


    function getCurrentRound() public view returns (uint) {
        return currentRound;
    }

    function getParticipantIds() public view returns (string[] memory) {
        return participantIds;
    }

    function setModelUpdateWindowSize(uint _windowSize) public {
        require(_windowSize > 0, "Window size must be greater than 0.");
        modelUpdateWindowSize = _windowSize;
        emit ModelUpdateWindowSizeSet(_windowSize);
    }

    function getModelUpdateWindowSize() public view returns (uint) {
        return modelUpdateWindowSize;
    }
}