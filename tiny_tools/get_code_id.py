from datetime import datetime
import subprocess


def get_latest_git_commit_id():
    try:
        # 使用git命令获取最新commit的ID
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        # 将输出从字节串转换为字符串，并去掉末尾的换行符
        commit_id = commit_id.decode('utf-8').strip()
        return commit_id[:5]
    except subprocess.CalledProcessError as e:
        # 如果git命令执行出错，可以在这里处理错误
        print(f"Error: {e}")
        return None
    except Exception as e:
        # 捕获其他可能的异常
        print(f"An unexpected error occurred: {e}")
        return None


# 调用函数并打印结果
latest_commit_id = get_latest_git_commit_id()
if latest_commit_id:
    print(f"Latest Git Commit ID: {latest_commit_id}")
else:
    raise ValueError("Failed to retrieve the latest Git commit ID.")


def generate_datetime_string():
    now = datetime.now()
    formatted_datetime = now.strftime('%Y-%m-%d-%H-%M')
    return formatted_datetime


# 使用函数
date_time_str = generate_datetime_string()

if __name__ == '__main__':
    print(latest_commit_id)
    print(date_time_str)
