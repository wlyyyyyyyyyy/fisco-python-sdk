#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
from bcos3sdk.bcos3client import Bcos3Client
from client.contractnote import ContractNote
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client_config import client_config
import os
import traceback

# ----- 合约信息 -----
CONTRACT_NAME = "HelloWorld"
CONTRACT_PATH = "./contracts/HelloWorld.sol"  # 假设合约文件和脚本在同一目录下，或者根据你的实际路径修改
ABI_PATH = "./contracts/HelloWorld.abi"
BIN_PATH = "./contracts/HelloWorld.bin"
CONTRACT_NOTE_NAME = "hello_world_note"
CONTRACT_ADDRESS = ""  # 部署后会自动更新，或者你也可以手动填写已部署的合约地址

# ----- 配置 -----
demo_config = client_config

# ----- 自动编译合约 (如果 ABI 和 BIN 文件不存在) -----
if not (os.path.exists(ABI_PATH) and os.path.exists(BIN_PATH)):
    print(f"ABI or BIN file not found, compiling contract: {CONTRACT_PATH}")
    Compiler.compile_file(CONTRACT_PATH)

abi_file = ABI_PATH
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

try:
    client = Bcos3Client()
    print(client.getinfo())

    # ========== 部署合约 ==========
    if not CONTRACT_ADDRESS:
        print("\n>>Deploying contract:----------------------------------------------------------")
        with open(BIN_PATH, 'r') as f:
            contract_bin = f.read()
            f.close()
        deploy_result = client.deploy(contract_bin)
        if deploy_result is None or deploy_result["status"] != 0:
            print(f"Deploy contract failed, result: {deploy_result}")
            exit()
        CONTRACT_ADDRESS = deploy_result["contractAddress"]
        print(f"Deploy contract success, contract address: {CONTRACT_ADDRESS}")
        ContractNote.save_address_to_contract_note(CONTRACT_NOTE_NAME, CONTRACT_NAME, CONTRACT_ADDRESS)
    else:
        print(f"\n>>Using existing contract at: {CONTRACT_ADDRESS}")

    # ========== 调用合约的 get 函数 ==========
    # ========== 调用合约的 get 函数 ==========
    # ========== 调用合约的 get 函数 ==========
    def get_name():
        print("\n>>Calling get() function:----------------------------------------------------------")
        to_address = CONTRACT_ADDRESS
        fn_name = "get"
        result = client.call(to_address, contract_abi, fn_name, [])
        if result is not None:  # 简单判断 result 是否为空
            name = result[0]  # 直接取元组的第一个元素作为 name
            print(f"Current name from contract: {name}")
            return name
        else:
            print(f"Call get() failed, result: {result}") # 仍然保留失败时的打印信息，但现在只有在 result 为 None 时才会执行
            return None

    # ========== 调用合约的 set 函数 ==========
    def set_name(new_name):
        print("\n>>Calling set() function:----------------------------------------------------------")
        to_address = CONTRACT_ADDRESS
        fn_name = "set"
        args = [new_name]
        receipt = client.sendRawTransaction(to_address, contract_abi, fn_name, args)
        if receipt is not None and receipt["status"] == 0:
            print(f"Set name transaction success, receipt: {receipt}")
            return True
        else:
            print(f"Set name transaction failed, receipt: {receipt}")
            return False

    # ========== 客户端主流程 ==========
    print("\n>>Starting HelloWorld Client Demo:----------------------------------------------------------")

    current_name = get_name()  # 获取当前名字

    if current_name:
        new_name = input("\nEnter a new name to set in the contract: ")
        if new_name:
            if set_name(new_name):  # 设置新的名字
                print("\n>>Name updated successfully!")
                get_name()  # 再次获取并打印新的名字
            else:
                print("\n>>Failed to update name.")
        else:
            print("\n>>No new name provided, exiting.")
    else:
        print("\n>>Could not retrieve current name, exiting.")

    client.finish()
    print("\n>>HelloWorld Client Demo finished.----------------------------------------------------------")

except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()
    client.finish()
    sys.exit(1)

sys.exit(0)