pragma solidity ^0.7.6;

contract EvenSimplerFederatedLearning {
    string private globalModel;

    constructor() {
        // 初始模型可以设置为空字符串或者一个默认值
        globalModel = "Initial Model - Very Simple";
    }

    function updateModel(string memory updatedModel) public {
        // 任何人都可以更新模型 (极简示例，不考虑权限控制)
        globalModel = updatedModel;
    }

    function getModel() public view returns (string memory) {
        return globalModel;
    }
}