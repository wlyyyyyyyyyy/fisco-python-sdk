import os

# 模板文件路径 (假设你的原始 bcos3_sdk_config.ini 文件是模板)
template_file = "./bcos3sdklib/bcos3_sdk_config.ini.template"  # 你需要将你的 bcos3_sdk_config.ini 重命名为 .template

# 输出目录
output_dir = "./bcos3sdklib"

# Trainer 节点数量
num_trainers = 3

# Keystore 文件路径前缀 (假设 keystore 文件都放在 ./bin/accounts/clientX.keystore)
keystore_path_prefix = "./bin/accounts/client"

#  假设你的 bcos3_sdk_config.ini.template 文件中，keystore 路径使用了占位符 {{keystore_path}}

def generate_trainer_config(trainer_id):
    """为指定的 Trainer ID 生成配置文件"""
    output_file = os.path.join(output_dir, f"bcos3_sdk_config_trainer{trainer_id}.ini")
    keystore_file = f"{keystore_path_prefix}{trainer_id}.keystore"

    with open(template_file, "r") as infile:
        template_content = infile.read()

    # 替换占位符 (这里只替换了 keystore_path, 你可以根据需要添加更多替换逻辑)
    config_content = template_content.replace("{{keystore_path}}", keystore_file)

    with open(output_file, "w") as outfile:
        outfile.write(config_content)
    print(f"已生成配置文件: {output_file}")

if __name__ == "__main__":
    for i in range(1, num_trainers + 1):
        generate_trainer_config(i)

    print("\n配置文件生成完成！")