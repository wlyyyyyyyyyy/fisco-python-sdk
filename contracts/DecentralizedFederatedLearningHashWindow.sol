pragma solidity ^0.7.6;

contract DecentralizedFederatedLearningHashWindow {
    uint256 public numParticipants;
    uint256 public windowSize;
    uint256 public currentRound;
    mapping(address => bool) public participants;
    mapping(uint256 => address) public roundParticipants;
    mapping(uint256 => mapping(address => string)) public participantModelUpdates;
    mapping(uint256 => mapping(address => string)) public localGlobalModelHashes;
    address public deployer;

    constructor() {
        deployer = msg.sender;
        currentRound = 0;
    }

    modifier onlyDeployer() {
        require(msg.sender == deployer, "Only the deployer can call this function.");
        _;
    }

    modifier onlyRegisteredParticipant() {
        require(participants[msg.sender], "Only registered participants can call this function.");
        _;
    }

    function setNumParticipants(uint256 _numParticipants) public onlyDeployer {
        require(currentRound == 0, "Cannot set numParticipants after rounds have started.");
        numParticipants = _numParticipants;
    }

    function setWindowSize(uint256 _windowSize) public onlyDeployer {
        require(currentRound == 0, "Cannot set windowSize after rounds have started.");
        windowSize = _windowSize;
    }

    function registerParticipant(address _participant) public onlyDeployer {
        require(!participants[_participant], "Participant is already registered.");
        participants[_participant] = true;
    }

    function uploadModelUpdate(uint256 _round, string memory _modelUpdate) public onlyRegisteredParticipant {
        require(_round == currentRound, "Cannot upload update for a different round.");
        participantModelUpdates[_round][msg.sender] = _modelUpdate;
        // Add participant to the current round's list if not already there
        bool found = false;
        for (uint i = 0; i < roundParticipants[_round].length; i++) {
            if (roundParticipants[_round][i] == msg.sender) {
                found = true;
                break;
            }
        }
        if (!found) {
            roundParticipants[_round].push(msg.sender);
        }
    }

    function submitLocalGlobalModelHash(uint256 _round, string memory _modelHash) public onlyRegisteredParticipant {
        require(_round == currentRound, "Cannot submit hash for a different round.");
        localGlobalModelHashes[_round][msg.sender] = _modelHash;
    }

    function startNextRound() public onlyDeployer {
        require(roundParticipants[currentRound].length == numParticipants, "Not all registered participants have submitted their updates for the current round.");
        // Check if all submitted hashes are the same (optional, can be done off-chain)
        if (numParticipants > 0) {
            string memory firstHash = localGlobalModelHashes[currentRound][roundParticipants[currentRound][0]];
            for (uint i = 1; i < roundParticipants[currentRound].length; i++) {
                require(keccak256(bytes(localGlobalModelHashes[currentRound][roundParticipants[currentRound][i]])) == keccak256(bytes(firstHash)), "Local global model hashes are not the same.");
            }
        }
        currentRound++;
        // No need to clear roundParticipants or hashes, they are managed by the round number
    }

    function getParticipantUpdates(uint256 _round) public view returns (string memory) {
        require(_round < currentRound, "Cannot get updates for the current or future round.");
        string memory updates = "[";
        bool first = true;
        for (uint i = 0; i < roundParticipants[_round].length; i++) {
            address participant = roundParticipants[_round][i];
            if (!first) {
                updates = string(abi.encodePacked(updates, ","));
            }
            updates = string(abi.encodePacked(updates, '{"participantId":"', addressToString(participant), '","modelUpdate":"', participantModelUpdates[_round][participant], '"}'));
            first = false;
        }
        updates = string(abi.encodePacked(updates, "]"));
        return updates;
    }

    function getAllLocalGlobalModelHashes(uint256 _round) public view returns (stringmemory) {
        require(_round == currentRound, "Cannot get hashes for the current round.");
        uint256 numSubmitted = 0;
        for (uint i = 0; i < roundParticipants[_round].length; i++) {
            if (bytes(localGlobalModelHashes[_round][roundParticipants[_round][i]]).length > 0) {
                numSubmitted++;
            }
        }
        stringmemory hashes = new string(numSubmitted);
        uint256 index = 0;
        for (uint i = 0; i < roundParticipants[_round].length; i++) {
            string memory hash = localGlobalModelHashes[_round][roundParticipants[_round][i]];
            if (bytes(hash).length > 0) {
                hashes[index] = hash;
                index++;
            }
        }
        return hashes;
    }

    function isParticipantRegistered(address _participant) public view returns (bool) {
        return participants[_participant];
    }

    // Helper function to convert address to string
    function addressToString(address _addr) internal pure returns (string memory) {
        bytes memory b = abi.encodePacked(uint160(_addr));
        string memory s = "";
        for (uint i = 0; i < b.length; i++) {
            uint8 byte = uint8(b[i]);
            uint8 hi = byte >> 4;
            uint8 lo = byte & 0x0f;
            s = string(abi.encodePacked(s, char(hi > 9 ? hi + 0x37 : hi + 0x30), char(lo > 9 ? lo + 0x37 : lo + 0x30)));
        }
        return s;
    }
}