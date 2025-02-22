#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import traceback
import copy  # 导入 copy 模块，用于深拷贝 ClientConfig 对象
from bcos3sdk.bcos3client import Bcos3Client
from client.common.compiler import Compiler
from client.datatype_parser import DatatypeParser
from client_config import client_config as ClientConfig
from eth_utils import to_checksum_address # 导入 to_checksum_address


def deploy_inter_client_message_contract(deploy_client):
    """使用部署客户端部署 InterClientMessageContract 合约"""
    print(">> 使用客户端 1 (部署者 - Trainer 1) 部署 InterClientMessageContract 合约...") #  更清晰的日志
    contract_file = "./contracts/InterClientMessageContract.sol" #  新的合约文件
    Compiler.compile_file(contract_file)
    bin_file = "./contracts/InterClientMessageContract.bin"
    with open(bin_file, 'r') as f:
        contract_bin = f.read().strip()
    result = deploy_client.deploy(contract_bin)
    return result

def send_message_to_client(message_client, contract_address, contract_abi, recipient_address, message_text):
    """使用消息客户端向指定接收者地址发送消息"""
    print(f">> {message_client.getinfo()} 向 {recipient_address} 发送消息...") #  更详细的日志
    receipt = message_client.sendRawTransaction(contract_address, contract_abi,
                                               "sendMessage", [recipient_address, message_text]) #  调用 sendMessage 函数
    return receipt

def get_client_messages(message_client, contract_address, contract_abi):
    """使用消息客户端获取发送给自己的所有消息"""
    print(f">> {message_client.getinfo()} 获取消息...") # 更详细的日志
    result = message_client.call(contract_address, contract_abi, "getMessages") # 调用 getMessages 函数
    message_list = data_parser.parse_output("getMessages", result['output'])[0] #  解析返回的消息列表
    return message_list


def main():
    client1 = None
    client2 = None
    client3 = None
    try:
        # 创建默认配置对象 (加载 bcos3_sdk_config.ini)
        default_config = ClientConfig()

        # ======  客户端 1 (部署者 - Trainer 1) ======
        config1 = ClientConfig(config_file="bcos3sdklib/bcos3_sdk_config_trainer1.ini") # 指定 trainer1 的配置文件
        client1 = Bcos3Client(config1)
        print("\n======  客户端 1 (部署者 - Trainer 1)  ======") #  更清晰的客户端标识
        client1_info_str = client1.getinfo() # Get getinfo() output as string
        print("客户端 1 info:", client1_info_str) # Print the string output

        # Extract account address from client1_info_str
        account_prefix = "account:["
        account_start_index = client1_info_str.find(account_prefix)
        if account_start_index != -1:
            account_start_index += len(account_prefix)
            account_end_index = client1_info_str.find("]", account_start_index)
            if account_end_index != -1:
                client1_account_address_str = client1_info_str[account_start_index:account_end_index]
                client1_address = to_checksum_address(client1_account_address_str)
                print(f"Extracted Client 1 address: {client1_address}") # Print extracted address
            else:
                raise Exception("Could not find account end bracket ']' in client1.getinfo() output")
        else:
            raise Exception("Could not find 'account:' prefix in client1.getinfo() output")


        # 部署 InterClientMessageContract 合约 (使用客户端 1)
        deploy_result = deploy_inter_client_message_contract(client1) #  部署新的合约
        contract_address = deploy_result.get("contractAddress")
        if not contract_address:
            print("客户端 1 部署合约失败!")
            return
        print("客户端 1 部署 InterClientMessageContract 成功，合约地址:", contract_address)

        # ======  客户端 2 (消息客户端 - Trainer 2) ======
        config2 = ClientConfig(config_file="bcos3sdklib/bcos3_sdk_config_trainer2.ini") # 指定 trainer2 的配置文件
        client2 = Bcos3Client(config2)
        print("\n======  客户端 2 (消息客户端 - Trainer 2)  ======") # 更清晰的客户端标识
        client2_info_str = client2.getinfo() # Get getinfo() output as string
        print("客户端 2 info:", client2_info_str) # Print the string output

        # Extract account address from client2_info_str
        account_start_index = client2_info_str.find(account_prefix)
        if account_start_index != -1:
            account_start_index += len(account_prefix)
            account_end_index = client2_info_str.find("]", account_start_index)
            if account_end_index != -1:
                client2_account_address_str = client2_info_str[account_start_index:account_end_index]
                client2_address = to_checksum_address(client2_account_address_str)
                print(f"Extracted Client 2 address: {client2_address}") # Print extracted address
            else:
                raise Exception("Could not find account end bracket ']' in client2.getinfo() output")
        else:
            raise Exception("Could not find 'account:' prefix in client2.getinfo() output")


        # ======  客户端 3 (消息客户端 - Trainer 3) ======  添加客户端 3
        config3 = ClientConfig(config_file="bcos3sdklib/bcos3_sdk_config_trainer3.ini") # 指定 trainer3 的配置文件
        client3 = Bcos3Client(config3)
        print("\n======  客户端 3 (消息客户端 - Trainer 3)  ======") # 更清晰的客户端标识
        client3_info_str = client3.getinfo() # Get getinfo() output as string
        print("客户端 3 info:", client3_info_str) # Print the string output

        # Extract account address from client3_info_str
        account_start_index = client3_info_str.find(account_prefix)
        if account_start_index != -1:
            account_start_index += len(account_prefix)
            account_end_index = client3_info_str.find("]", account_start_index)
            if account_end_index != -1:
                client3_account_address_str = client3_info_str[account_start_index:account_end_index]
                client3_address = to_checksum_address(client3_account_address_str)
                print(f"Extracted Client 3 address: {client3_address}") # Print extracted address
            else:
                raise Exception("Could not find account end bracket ']' in client3.getinfo() output")
        else:
            raise Exception("Could not find 'account:' prefix in client3.getinfo() output")


        # 加载合约ABI文件 (使用 InterClientMessageContract.abi)
        abi_file = "./contracts/InterClientMessageContract.abi" #  新的 ABI 文件
        global data_parser
        data_parser = DatatypeParser()
        print("DatatypeParser 初始化完成")  # 调试信息
        data_parser.load_abi_file(abi_file)
        print(f"ABI 文件加载完成，abi_file = {abi_file}") # 调试信息
        contract_abi = data_parser.contract_abi
        print(f"contract_abi 赋值完成，contract_abi = {contract_abi}") # 调试信息


        print("\n--- 客户端地址 ---") #  打印客户端地址信息
        print(f"客户端 1 地址 (Trainer 1): {client1_address}") #  更清晰的客户端标识
        print(f"客户端 2 地址 (Trainer 2): {client2_address}") #  更清晰的客户端标识
        print(f"客户端 3 地址 (Trainer 3): {client3_address}") #  更清晰的客户端标识
        print("---  ---------- ---")


        #  客户端 1 向 客户端 2 和 客户端 3 发送消息
        send_message_to_client(client1, contract_address, contract_abi, client2_address, "Hello Client 2 from Client 1!") #  客户端 1 发送给 客户端 2
        send_message_to_client(client1, contract_address, contract_abi, client3_address, "Hi Client 3, this is Client 1!") #  客户端 1 发送给 客户端 3

        #  客户端 2 向 客户端 1 和 客户端 3 发送消息
        send_message_to_client(client2, contract_address, contract_abi, client1_address, "Message from Client 2 to Client 1.") # 客户端 2 发送给 客户端 1
        send_message_to_client(client2, contract_address, contract_abi, client3_address, "Client 2 says hello to Client 3!") # 客户端 2 发送给 客户端 3

        #  客户端 3 向 客户端 1 和 客户端 2 发送消息
        send_message_to_client(client3, contract_address, contract_abi, client1_address, "Client 3 to Client 1: Greetings!") # 客户端 3 发送给 客户端 1
        send_message_to_client(client3, contract_address, contract_abi, client2_address, "Hello Client 2, Client 3 here.") # 客户端 3 发送给 客户端 2


        print("\n--- 获取客户端消息 ---") #  开始获取消息

        #  客户端 1 获取消息
        messages_for_client1 = get_client_messages(client1, contract_address, contract_abi) #  客户端 1 获取自己的消息
        print(f"\n客户端 1 收到的消息 (Trainer 1 - {client1.getinfo()}):") # 打印客户端 1 收到的消息，包含 Trainer 标识
        if messages_for_client1: #  如果消息列表不为空
            for msg in messages_for_client1: # 遍历消息列表
                print(f"- {msg}") # 打印每条消息
        else:
            print("  (No messages)") #  如果没有消息，则提示

        #  客户端 2 获取消息
        messages_for_client2 = get_client_messages(client2, contract_address, contract_abi) # 客户端 2 获取自己的消息
        print(f"\n客户端 2 收到的消息 (Trainer 2 - {client2.getinfo()}):") # 打印客户端 2 收到的消息，包含 Trainer 标识
        if messages_for_client2: # 如果消息列表不为空
            for msg in messages_for_client2: # 遍历消息列表
                print(f"- {msg}") # 打印每条消息
        else:
            print("  (No messages)") # 如果没有消息，则提示

        #  客户端 3 获取消息
        messages_for_client3 = get_client_messages(client3, contract_address, contract_abi) # 客户端 3 获取自己的消息
        print(f"\n客户端 3 收到的消息 (Trainer 3 - {client3.getinfo()}):") # 打印客户端 3 收到的消息，包含 Trainer 标识
        if messages_for_client3: # 如果消息列表不为空
            for msg in messages_for_client3: # 遍历消息列表
                print(f"- {msg}") # 打印每条消息
        else:
            print("  (No messages)") # 如果没有消息，则提示

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
    finally:
        if client1:
            client1.finish()
        if client2:
            client2.finish()
        if client3:
            client3.finish()

if __name__ == '__main__':
    main()