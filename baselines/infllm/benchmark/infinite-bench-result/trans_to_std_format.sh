#!/bin/bash

# 设置路径，假设path变量是你需要操作的目录路径
path="."

# 创建目标文件夹
destination="$path/yi-34b-200k"
mkdir -p "$destination"

# 查找所有以jsonl结尾的文件并进行操作
find "$path" -type f -name "*.jsonl" | while read file; do
    # 获取文件名
    filename=$(basename "$file")

    # 复制文件到目标文件夹
    cp "$file" "$destination/preds_$filename"

    # 替换 "pred" 为 "prediction" 和 "answers" 为 "ground_truth"
    sed -i 's/\bpred\b/prediction/g' "$destination/preds_$filename"
    sed -i 's/\banswers\b/ground_truth/g' "$destination/preds_$filename"
done

echo "操作完成，所有文件已复制并替换完成。"
