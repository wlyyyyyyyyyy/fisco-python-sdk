// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// 简化版的 LayerZero Endpoint 合约
contract LayerZeroEndpoint {

    // 另一条链的 Endpoint 地址
    address public otherChainEndpoint;

    // Oracle 地址
    address public oracle;

    // 存储已验证的区块头 (blockHash => blockNumber)
    mapping(bytes32 => uint256) public confirmedBlockHeaders;
    uint public lastConfirmedBlockNumber;

    // 消息 nonce (用于防止重放攻击)
    mapping(address => uint256) public nonces;

    // 事件：发送消息
    event MessageSent(address indexed sender, uint256 nonce, bytes message);

    // 事件：接收消息
    event MessageReceived(address indexed sender, uint256 nonce, bytes message);

    // 构造函数
    constructor(address _otherChainEndpoint, address _oracle) {
        otherChainEndpoint = _otherChainEndpoint;
        oracle = _oracle;
    }

    // 设置另一条链的 Endpoint 地址 (仅限 owner)
    function setOtherChainEndpoint(address _otherChainEndpoint) public onlyOwner {
        otherChainEndpoint = _otherChainEndpoint;
    }

    // 设置 Oracle 地址 (仅限 owner)
    function setOracle(address _oracle) public onlyOwner {
        oracle = _oracle;
    }

    // 发送消息
    function send(bytes calldata _message) external {
        uint256 nonce = nonces[msg.sender]++;
        emit MessageSent(msg.sender, nonce, _message);

        // (在实际的 LayerZero 中，这里会调用 Relayer 来提交交易证明)
    }

    // 接收消息 (由 Oracle 调用)
    // _srcChainId 参数在我们的简化版本中未使用，但在实际的 LayerZero 中用于标识源链
    function receive(
        address _sender,
        uint256 _nonce,
        bytes calldata _message,
        bytes32 _blockHash,
        uint _proof
    ) external {
        // 验证调用者是 Oracle
        require(msg.sender == oracle, "Only Oracle can call this function");

        // 验证区块头
        require(confirmedBlockHeaders[_blockHash] > 0, "Block header not confirmed");

        // 防止重放攻击
        require(_nonce == nonces[_sender], "Invalid nonce");
        nonces[_sender]++;
        // 验证交易包含证明 (简化版本，不做具体验证)
        require(_proof > 0, "Proof is needed");

        emit MessageReceived(_sender, _nonce, _message);
    }

    // 确认区块头 (由 Oracle 调用)
    function confirmBlockHeader(bytes32 _blockHash, uint256 _blockNumber) external {
        // 验证调用者是 Oracle
        require(msg.sender == oracle, "Only Oracle can call this function");
        // 检查区块高度是否连续
        require(_blockNumber == lastConfirmedBlockNumber + 1, "invalid blockNumber");

        confirmedBlockHeaders[_blockHash] = _blockNumber;
        lastConfirmedBlockNumber = _blockNumber;
    }

    // 模拟获取区块哈希 (实际应用中需要从链上获取)
    function getBlockHash(uint256 _blockNumber) public view returns (bytes32) {
        // 在这里实现从链上获取区块哈希的逻辑
        // (简化版本，直接返回一个随机值)
        return keccak256(abi.encodePacked(_blockNumber));
    }

      // 模拟获取区块高度
    function getBlockNumber() public view returns(uint) {
        // 在这里实现从链上获取最新区块高度的逻辑
        return lastConfirmedBlockNumber + 1;
    }

    // 辅助函数：仅限 owner 调用
    modifier onlyOwner() {
        // 在实际应用中，需要实现 owner 权限控制
        // (简化版本，不做任何检查)
        _;
    }
}