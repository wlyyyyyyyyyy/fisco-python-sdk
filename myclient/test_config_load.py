#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import traceback
from bcos3sdk.bcos3client import Bcos3Client
from client_config import client_config as ClientConfig

def main():
    client = None
    try:
        # 指定测试配置文件的路径 (请根据您的实际路径修改)
        config_file_path = "bcos3sdklib/test_sdk_config.ini"

        # 创建 ClientConfig 对象，加载测试配置文件
        config = ClientConfig(config_file=config_file_path)
        print(config.account_password)
        print(config.account_keyfile)
        print(f"Loaded config file: {config_file_path}") # 打印加载的配置文件路径

        # 创建 Bcos3Client 对象
        client = Bcos3Client(config)
        print("Bcos3Client 初始化成功") # 打印初始化成功信息

        # 获取并打印客户端信息 (包括地址)
        client_info_str = client.getinfo()
        print("\n======  客户端信息  ======")
        print("客户端 info:", client_info_str)

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
    finally:
        if client:
            client.finish()

if __name__ == '__main__':
    main()