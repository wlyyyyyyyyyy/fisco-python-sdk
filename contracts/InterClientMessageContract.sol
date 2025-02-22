pragma solidity >=0.4.2;
pragma abicoder v2;  //  <---  添加了这一行，启用 ABI coder v2

contract InterClientMessageContract {

    // 使用 mapping 存储每个地址的消息列表
    mapping(address => string[]) public clientMessages;

    // 发送消息函数
    function sendMessage(address recipient, string memory message) public {
        // 将消息添加到接收者地址的消息列表中
        clientMessages[recipient].push(message);
    }

    // 获取消息函数
    function getMessages() public view returns (string[] memory) {
        // 返回发送给当前调用者 (msg.sender) 的所有消息
        return clientMessages[msg.sender];
    }
}