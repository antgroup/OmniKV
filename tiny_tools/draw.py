import matplotlib.pyplot as plt
import os
from tiny_tools.get_code_id import date_time_str


def plot_curve(data, file_path='debug_logs/imgs', name='loss', smooth_step=128):
    # 创建一个新的图形
    fig, ax = plt.subplots()
    data = [
        sum(data[l: l + smooth_step]) / smooth_step if l + smooth_step <= len(data) else sum(data[l:]) / len(data[l:])
        for l in range(0, len(data), smooth_step)]

    # 绘制曲线图
    ax.plot(data)

    # 设置图形标题和轴标签
    ax.set_title('Curve Plot')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, f"{name}_{date_time_str}.png"))

    # 关闭图形
    plt.close(fig)


def plot_curve_online(data):
    # 创建一个新的图形
    fig, ax = plt.subplots()

    # 绘制曲线图
    ax.plot(data)

    # 设置图形标题和轴标签
    ax.set_title('Curve Plot')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # 关闭图形
    plt.show()


if __name__ == '__main__':
    # plot_curve([0.2, 0.1, 0.3], './debug_logs/imgs', smooth_step=2, name='test')
    from tiny_tools.read_logs import read_log

    loss_list = [float(_.group(1)) for _ in
                 read_log('debug_logs/main_2024-06-03-09-34_195d7.log', r'% loss = ([0-9.]+)')]
    plot_curve(loss_list, name='smooth_loss_baseline', smooth_step=16)
    loss_list = [float(_.group(1)) for _ in
                 read_log('debug_logs/main_2024-06-03-11-13_b2fde.log', r'% loss = ([0-9.]+)')]
    plot_curve(loss_list, name='smooth_loss_ours', smooth_step=16)
