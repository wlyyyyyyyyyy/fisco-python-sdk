#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# 获取当前文件 (simple_test.py) 所在的目录 (mytest)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将 bcos3sdk 所在的目录添加到 sys.path
# 假设 bcos3sdk 在你的 python-sdk 目录下 (与 mytest 同级)
bcos3sdk_path = os.path.join(current_dir, "..", "bcos3sdk")
sys.path.insert(0, bcos3sdk_path)


from bcos3sdk.bcos3client import Bcos3Client
from .client_config import client_config  # 使用相对导入, 现在应该可以正常工作了
from eth_utils import to_checksum_address
import traceback

# 合约文件路径 (相对于 simple_test.py, 但在上一层目录运行)
CONTRACT_PATH = "./mytest/contracts/SimpleStorage.sol"
ABI_PATH = "./mytest/contracts/SimpleStorage.abi"
BIN_PATH = "./mytest/contracts/SimpleStorage.bin"

def main():
    try:
        # 初始化客户端
        client = Bcos3Client(client_config)
        print("Client info:", client.getinfo())

        # 部署合约
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
        deploy_result = client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print("Deploy contract failed:", deploy_result)
            sys.exit(1)

        contract_address = deploy_result["contractAddress"]
        print("Contract deployed at:", contract_address)

        # 加载 ABI
        with open(ABI_PATH, 'r') as f:
            contract_abi = f.read()

        # 设置值
        new_value = 12345
        fn_name = "setValue"
        args = [new_value]
        receipt = client.sendRawTransaction(contract_address, contract_abi, fn_name, args)
        if receipt['status'] != 0:
            print("setValue failed:", receipt)
            sys.exit(1)
        print("setValue receipt:", receipt)

        # 读取值
        fn_name = "value"
        args = []  # value() 函数没有参数
        result = client.call(contract_address, contract_abi, fn_name, args)
        print("getValue result:", result)
        ret_value = result[0]

        # 验证值
        if int(ret_value) == new_value:
            print("Test passed!")
        else:
            print("Test failed! Expected:", new_value, "Got:", ret_value)

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.finish()

if __name__ == "__main__":
    main()