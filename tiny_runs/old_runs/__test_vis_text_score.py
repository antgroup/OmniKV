import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_mat(list_of_word_score, current_word, width):
    # 获取所有单词和它们的分数
    words, scores = zip(*list_of_word_score)

    # 计算行数
    num_rows = (len(words) + width - 1) // width

    # 创建一个空矩阵来存储分数
    score_matrix = np.zeros((num_rows, width))
    word_matrix = [["" for _ in range(width)] for _ in range(num_rows)]

    for idx, (word, score) in enumerate(list_of_word_score):
        row = idx // width
        col = idx % width
        score_matrix[row, col] = score
        word_matrix[row][col] = word

    # 绘制热图
    plt.figure(figsize=(width * 2, num_rows * 1))
    sns.heatmap(score_matrix, annot=word_matrix, fmt='', cmap='coolwarm', cbar=True, linewidths=.5, linecolor=None)

    plt.title(f"Attention scores for '{current_word}'")
    plt.tight_layout()
    plt.axis('off')
    # plt.show()
    plt.savefig("debug_logs/vis_text.png")


# 测试代码
# list_of_word_score = [("word1", 0.1), ("word2", 0.3), ("word3", 0.5), ("word4", 0.2), ("word5", 0.4)]
list_of_word_score = [(str(i), 0.2) for i in range(100)]
current_word = "query_word"
width = 52

draw_mat(list_of_word_score, current_word, width)