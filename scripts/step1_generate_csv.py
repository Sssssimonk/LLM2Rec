"""
Step 1: Generate Training CSV Files
从已有的 downstream 评估数据读取序列和标题映射，生成训练 CSV 文件
"""
import json
import csv
from pathlib import Path

# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "llm2rec" / "data"

# 6个训练域
TRAINING_CATEGORIES = [
    "Video_Games",
    "Arts_Crafts_and_Sewing", 
    "Movies_and_TV",
    "Home_and_Kitchen",
    "Electronics",
    "Tools_and_Home_Improvement"
]


def load_downstream_data(category):
    """
    从 downstream 文件夹加载序列和标题映射
    
    重要: 论文明确说明 "Only the training data is used for pre-training LLM2Rec"
    因此使用 train_data.txt 而不是 data.txt (完整序列)
    """
    downstream_dir = DATA_DIR / category / "5-core" / "downstream"
    
    # 加载 item_titles.json (ID从1开始)
    with open(downstream_dir / "item_titles.json", 'r', encoding='utf-8') as f:
        item_titles = json.load(f)
    
    # 加载训练序列 - 使用 train_data.txt (论文要求)
    with open(downstream_dir / "train_data.txt", 'r', encoding='utf-8') as f:
        sequences = [list(map(int, line.strip().split())) for line in f if line.strip()]
    
    # 确保所有序列长度 <= 10 (论文要求)
    max_len = 10
    sequences = [seq[:max_len] for seq in sequences if len(seq) > 0]
    
    return sequences, item_titles


def write_csv_samples(sequences, item_titles, output_path):
    """
    流式生成并写入训练样本 - 完整扩展策略
    
    对每个序列,生成所有可能的 (history, target) 对:
    - 序列 [1,2,3,4] 生成 3 个样本:
      ([1], 2), ([1,2], 3), ([1,2,3], 4)
    
    这是标准的序列推荐数据扩展方法,不做采样
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['history_item_id', 'history_item_title', 'item_id', 'item_title'])
        writer.writeheader()
        
        for seq in sequences:
            # 遍历所有可能的位置(1到len(seq)-1) - 完整扩展
            for i in range(1, len(seq)):
                history_ids = seq[:i]
                target_id = seq[i]
                
                # 获取标题(ID在JSON中是字符串key)
                history_titles = [item_titles[str(item_id)] for item_id in history_ids]
                target_title = item_titles[str(target_id)]
                
                writer.writerow({
                    'history_item_id': repr(history_ids),  # Python list字符串表示
                    'history_item_title': repr(history_titles),
                    'item_id': target_id,
                    'item_title': target_title
                })
                count += 1
    
    print(f"  [OK] 生成 {count} 个样本 -> {output_path}")
    return count


def main():
    import random
    random.seed(42)  # 固定随机种子保证可重现
    
    print("=" * 60)
    print("Step 1: 生成训练 CSV 文件")
    print("=" * 60)
    
    # 1. 为每个训练域生成单域 CSV
    domain_data = []
    for category in TRAINING_CATEGORIES:
        print(f"\n处理 {category}...")
        
        sequences, item_titles = load_downstream_data(category)
        
        # 保存单域训练CSV (完整扩展所有位置)
        output_path = DATA_DIR / category / "5-core" / "train" / f"{category}_5_mixed.csv"
        count = write_csv_samples(sequences, item_titles, output_path)
        
        # 记录域信息用于后续混合
        domain_data.append((sequences, item_titles, count))
    
    # 2. 生成 AmazonMix-6 混合数据集 (train/valid 划分)
    print(f"\n生成 AmazonMix-6 混合数据集...")
    
    total_samples = sum(count for _, _, count in domain_data)
    split_idx = int(total_samples * 0.9)
    print(f"  总样本数: {total_samples}, 训练集: {split_idx}, 验证集: {total_samples - split_idx}")
    
    # 准备输出路径
    train_path = DATA_DIR / "AmazonMix-6" / "5-core" / "train" / "AmazonMix-6_5_mixed.csv"
    valid_path = DATA_DIR / "AmazonMix-6" / "5-core" / "valid" / "AmazonMix-6_5_mixed.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 流式写入train和valid (完整扩展)
    train_count = 0
    valid_count = 0
    current_idx = 0
    
    train_file = open(train_path, 'w', encoding='utf-8', newline='')
    valid_file = open(valid_path, 'w', encoding='utf-8', newline='')
    
    train_writer = csv.DictWriter(train_file, fieldnames=['history_item_id', 'history_item_title', 'item_id', 'item_title'])
    valid_writer = csv.DictWriter(valid_file, fieldnames=['history_item_id', 'history_item_title', 'item_id', 'item_title'])
    
    train_writer.writeheader()
    valid_writer.writeheader()
    
    for sequences, item_titles, _ in domain_data:
        for seq in sequences:
            # 完整扩展所有位置
            for i in range(1, len(seq)):
                history_ids = seq[:i]
                target_id = seq[i]
                
                history_titles = [item_titles[str(item_id)] for item_id in history_ids]
                target_title = item_titles[str(target_id)]
                
                row = {
                    'history_item_id': repr(history_ids),
                    'history_item_title': repr(history_titles),
                    'item_id': target_id,
                    'item_title': target_title
                }
                
                # 根据索引决定写入train还是valid
                if current_idx < split_idx:
                    train_writer.writerow(row)
                    train_count += 1
                else:
                    valid_writer.writerow(row)
                    valid_count += 1
                
                current_idx += 1
    
    train_file.close()
    valid_file.close()
    
    print(f"  [OK] 训练集: {train_count} 个样本 -> {train_path}")
    print(f"  [OK] 验证集: {valid_count} 个样本 -> {valid_path}")
    
    print("\n" + "=" * 60)
    print("Step 1 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
