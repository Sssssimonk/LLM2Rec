"""
Step 2: Generate IEM Training Data
生成 IEM 阶段所需的数据:
1. item_titles.txt - 6个训练域的所有物品标题合并去重
2. training_item_pairs_gap24.jsonl - 用户序列中间隔<=24的物品标题对
"""
import json
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


def generate_item_titles_txt():
    """生成 item_titles.txt - 6个域所有物品标题合并去重"""
    print("\n生成 item_titles.txt...")
    
    all_titles = set()
    
    for category in TRAINING_CATEGORIES:
        # 从 downstream/item_titles.json 读取标题
        json_path = DATA_DIR / category / "5-core" / "downstream" / "item_titles.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            item_titles = json.load(f)
        
        # 添加到集合(自动去重)
        all_titles.update(item_titles.values())
        print(f"  {category}: {len(item_titles)} 个物品")
    
    # 排序后保存(保证可重现)
    sorted_titles = sorted(all_titles)
    
    output_path = DATA_DIR / "AmazonMix-6" / "5-core" / "info" / "item_titles.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for title in sorted_titles:
            f.write(title + '\n')
    
    print(f"  [OK] 总计 {len(sorted_titles)} 个去重标题 -> {output_path}")


def generate_item_pairs(category):
    """为单个类别生成 training_item_pairs_gap24.jsonl"""
    
    # 加载序列和标题映射
    downstream_dir = DATA_DIR / category / "5-core" / "downstream"
    
    with open(downstream_dir / "item_titles.json", 'r', encoding='utf-8') as f:
        item_titles = json.load(f)
    
    with open(downstream_dir / "data.txt", 'r', encoding='utf-8') as f:
        sequences = [list(map(int, line.strip().split())) for line in f if line.strip()]
    
    # 生成物品对
    pairs = []
    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i + 1, min(i + 25, len(seq))):  # 间隔 <= 24
                item_i = seq[i]
                item_j = seq[j]
                
                title_i = item_titles[str(item_i)]
                title_j = item_titles[str(item_j)]
                
                pairs.append([title_i, title_j])
    
    # 保存为JSON数组格式(不是逐行JSONL!)
    output_path = DATA_DIR / category / "training_item_pairs_gap24.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False)
    
    print(f"  [OK] {category}: {len(pairs)} 个物品对 -> {output_path}")


def main():
    print("=" * 60)
    print("Step 2: 生成 IEM 训练数据")
    print("=" * 60)
    
    # 1. 生成 item_titles.txt
    generate_item_titles_txt()
    
    # 2. 为每个训练域生成 training_item_pairs_gap24.jsonl
    print("\n生成 training_item_pairs_gap24.jsonl...")
    for category in TRAINING_CATEGORIES:
        generate_item_pairs(category)
    
    print("\n" + "=" * 60)
    print("Step 2 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
