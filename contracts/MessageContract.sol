// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

contract MessageContract {
    // 定义消息事件
    event MessageSent(address indexed from, address indexed to, string message);

    // 定义消息结构体
    struct Message {
        address from;
        address to;      // 新增接收方地址
        string message;
        uint256 timestamp;
    }

    // 存储每个账户收到的消息
    mapping(address => Message[]) private messages;

    // 发送消息给指定账户
    function sendMessage(address _to, string calldata _message) external {
        // 将接收方地址和消息存入存储
        messages[_to].push(Message({
            from: msg.sender,  // 发送者
            to: _to,           // 接收者
            message: _message,
            timestamp: block.timestamp
        }));
        emit MessageSent(msg.sender, _to, _message);  // 事件中记录发送者和接收者
    }

    // 获取某账户收到的消息数量
    function getMessageCount(address _account) external view returns (uint256) {
        return messages[_account].length;
    }

    // 根据索引获取某账户收到的消息
    function getMessageByIndex(address _account, uint256 index) external view returns (address, address, string memory, uint256) {
        require(index < messages[_account].length, "Index out of range");
        Message storage msgData = messages[_account][index];
        return (msgData.from, msgData.to, msgData.message, msgData.timestamp);  // 返回发送者、接收者、消息和时间戳
    }
}
