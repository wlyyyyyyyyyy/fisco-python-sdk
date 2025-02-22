pragma solidity ^0.7.6;

contract HelloWorld {
    string private name;
    //event Onset(string newname);
    constructor() {
        name = "Hello, World!";
    }

    function get() public view returns (string memory) {
        return name;
    }

    function set(string memory n) public {
        //emit Onset(n);
        name = n;
    }
}
