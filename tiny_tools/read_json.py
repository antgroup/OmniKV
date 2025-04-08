import json
import hashlib


def read_config(config_path):
    """从JSON文件中读取配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def hash_dict_with_md5(dictionary):
    # 使用json.dumps将字典转换为字符串，并且
    # 通过sort_keys参数保证键的顺序是固定的
    dict_string = json.dumps(dictionary, sort_keys=True).encode('utf-8')

    # 创建一个MD5的哈希对象
    hasher = hashlib.md5()

    # 将序列化后的字符串传入该哈希对象
    hasher.update(dict_string)

    # 返回16进制的哈希摘要
    return hasher.hexdigest()


def read_dataset_in_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as _in:
        for line in _in:
            dt = json.loads(line)
            data += [dt]
    return data
