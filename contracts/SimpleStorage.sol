// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.7.6; // Using a more recent Solidity version

contract SimpleStorage {
    uint256 private storedData; // State variable to store a number
    address private lastSender; // Store the address of the last updater

    event DataStored(uint256 indexed newData, address indexed sender); // Event emitted when data is updated

    constructor() {
        // Optional: Initialize with a default value
        storedData = 0;
        lastSender = address(0); // Initialize with zero address
    }

    // Function to update the stored data
    function set(uint256 x) public {
        storedData = x;
        lastSender = msg.sender; // Record who called this function
        emit DataStored(x, msg.sender); // Emit an event
    }

    // Function to retrieve the stored data (read-only)
    function get() public view returns (uint256) {
        return storedData;
    }

    // Function to retrieve the address of the last sender (read-only)
    function getLastSender() public view returns (address) {
        return lastSender;
    }
}