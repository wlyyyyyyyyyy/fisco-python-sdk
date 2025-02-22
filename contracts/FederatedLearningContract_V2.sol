pragma solidity ^0.7.6;

contract FederatedLearningContract_V2 {
    string private globalModel;

    constructor() {
        // 初始模型
        globalModel = "Initial Model - V2";
    }

    // 下载模型函数 -  明确命名为 downloadModel
    function downloadModel() public view returns (string memory) {
        return globalModel;
    }

    // 上传模型函数 -  明确命名为 uploadModel
    function uploadModel(string memory updatedModel) public {
        globalModel = updatedModel;
    }
}