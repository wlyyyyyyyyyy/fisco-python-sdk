// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

contract SendAndGet {
    string private message;

    constructor(){
        message = "No message yet."; // Initialize in constructor
    }

    function sendData(string memory newMessage) public {
        message = newMessage;
    }

    function getData() public view returns (string memory) {
        return message;
    }
}