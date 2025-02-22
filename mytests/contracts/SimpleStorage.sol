// SPDX-License-Identifier: MIT
pragma solidity >=0.7.6 <0.9.0;

contract SimpleStorage {
    uint256 public value;

    function setValue(uint256 _newValue) public {
        value = _newValue;
    }
}