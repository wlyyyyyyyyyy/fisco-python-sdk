import sys
sys.path.append("./")
from client.stattool import StatTool
from bcos3sdk.bcos3client import Bcos3Client
from client.contractnote import ContractNote
from client.bcosclient import BcosClient
import os
from eth_utils import to_checksum_address
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client.bcoserror import BcosException, BcosError
from client_config import client_config

import traceback

# 从文件加载abi定义
demo_config = client_config
if os.path.isfile(demo_config.solc_path) or os.path.isfile(demo_config.solcjs_path):
    Compiler.compile_file("./contracts/SendAndGet.sol")

abi_file = "./contracts/SendAndGet.abi"
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

try:
    stat = StatTool.begin()
    client = Bcos3Client() # 初始化 Bcos3Client
    print(client.getinfo())

    # 部署合约
    print("\n>>Deploy:----------------------------------------------------------")
    with open("./contracts/SendAndGet.bin", 'r') as load_f: # 确保路径正确
        contract_bin = load_f.read()
        load_f.close()
    result = client.deploy(contract_bin)
    print("deploy", result)
    print("new address : ", result["contractAddress"])
    contract_name = os.path.splitext(os.path.basename(abi_file))[0]
    print("contract name: ", contract_name)
    memo = "tx:" + result["transactionHash"]
    # 把部署结果存入文件备查
    ContractNote.save_address_to_contract_note("demo",contract_name,
                                               result["contractAddress"])
    # 发送交易，调用 sendData 接口
    print("\n>>sendRawTransaction:----------------------------------------------------")
    to_address = result['contractAddress']  # use new deploy address
    data_to_send = "This is the data to send to SendAndGet contract!" # 要发送的数据
    args = [data_to_send]

    receipt = client.sendRawTransaction(to_address, contract_abi, "sendData", args) # 调用 sendData 函数
    print("receipt:", receipt)

    # 解析receipt里的log
    print("\n>>parse receipt and transaction:--------------------------------------")
    txhash = receipt['transactionHash']
    print("transaction hash: ", txhash)
    logresult = data_parser.parse_event_logs(receipt["logEntries"])
    i = 0
    for log in logresult:
        if 'eventname' in log:
            i = i + 1
            print("{}): log name: {} , data: {}".format(i, log['eventname'], log['eventdata']))
    # 获取对应的交易数据，解析出调用方法名和参数

    txresponse = client.getTransactionByHash(txhash)
    inputresult = data_parser.parse_transaction_input(txresponse['input'])
    print("transaction input parse:", txhash)
    print(inputresult)

    # 解析该交易在receipt里输出的output,即交易调用的方法的return值 (sendData 没有返回值，output 为空)
    outputresult = data_parser.parse_receipt_output(inputresult['name'], receipt['output'])
    print("receipt output :", outputresult)

    # 调用一下call，获取数据 getData
    print("\n>>Call:------------------------------------------------------------------------")
    res = client.call(to_address, contract_abi, "getData") # 调用 getData 函数
    print("call getData result:", res)

    stat.done()
    reqcount = next(client.request_counter)
    print("done,demo_tx,total request {},usedtime {},avgtime:{}".format(
            reqcount, stat.time_used, (stat.time_used / reqcount)
        ))

except BcosException as e:
    print("execute demo_transaction failed ,BcosException for: {}".format(e))
    traceback.print_exc()
except BcosError as e:
    print("execute demo_transaction failed ,BcosError for: {}".format(e))
    traceback.print_exc()
except Exception as e:
    client.finish()
    traceback.print_exc()
client.finish()
sys.exit(0)