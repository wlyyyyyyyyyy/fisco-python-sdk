// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;
contract HelloWorld6{
    string name;
    event onset(string newname);
    constructor(){
       name = "Hello, World!";
    }

    function get()  public view returns(string memory){
        return name;
    }

    function set(string memory n) public{
		emit onset(n);
    	name = n;
    }
}
