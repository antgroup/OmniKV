import logging
from tiny_tools.get_code_id import date_time_str, latest_commit_id
import os


def create_logger(logger_name, logfile, file_level=logging.INFO, terminal_level=logging.WARNING):
    # 创建Logger实例
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 创建Formatter，包括文件名和行号
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # 创建StreamHandler用于输出到终端
    if os.environ.get('DEBUG_LOG', False):
        terminal_level = logging.DEBUG
        file_level = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(terminal_level)
    stream_handler.setFormatter(formatter)

    # 创建FileHandler用于输出到文件
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # 给Logger实例添加Handler
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


logged_dict = {}


def warning_once(_logger, info):
    if info not in logged_dict:
        logged_dict[info] = 'ok'
        _logger.warning(info)


# 使用创建的Logger
base_path = 'debug_logs'
if os.environ.get("NO_NAS", False):
    base_path = '/jitai/debug_logs'
logger = create_logger('main', f'{base_path}/main_{date_time_str}_{latest_commit_id}.log')
logger_file_path = f'{base_path}/main_{date_time_str}_{latest_commit_id}.log'
logger.debug("初始化了logger")
