"""
Step 3: Verify Generated Data
校验所有生成的数据文件的格式正确性,打印统计摘要
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


def verify_csv(csv_path, category_name):
    """校验CSV文件格式"""
    if not csv_path.exists():
        print(f"  [FAIL] 缺失: {csv_path}")
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # 校验必要列
            required_cols = ['history_item_id', 'history_item_title', 'item_id', 'item_title']
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"  [FAIL] {category_name}: 缺少必要列")
                return False
            
            # 校验第一行能用eval解析
            if rows:
                first_row = rows[0]
                eval(first_row['history_item_id'])  # 应该是list
                eval(first_row['history_item_title'])  # 应该是list
            
            print(f"  [OK] {category_name}: {len(rows)} 个样本")
            return True
            
    except Exception as e:
        print(f"  [FAIL] {category_name}: 错误 - {e}")
        return False


def verify_item_titles_txt():
    """校验 item_titles.txt"""
    txt_path = DATA_DIR / "AmazonMix-6" / "5-core" / "info" / "item_titles.txt"
    
    if not txt_path.exists():
        print(f"  [FAIL] 缺失: {txt_path}")
        return False
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"  [OK] item_titles.txt: {len(lines)} 行")
        return True
        
    except Exception as e:
        print(f"  [FAIL] item_titles.txt: 错误 - {e}")
        return False


def verify_item_pairs(category):
    """校验 training_item_pairs_gap24.jsonl"""
    jsonl_path = DATA_DIR / category / "training_item_pairs_gap24.jsonl"
    
    if not jsonl_path.exists():
        print(f"  [FAIL] 缺失: {jsonl_path}")
        return False
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            pairs = json.loads(f.read().strip())  # 整体读取JSON数组
        
        # 校验格式
        if not isinstance(pairs, list):
            print(f"  [FAIL] {category}: 不是JSON数组格式")
            return False
        
        if pairs and not (isinstance(pairs[0], list) and len(pairs[0]) == 2):
            print(f"  [FAIL] {category}: 元素格式不正确")
            return False
        
        print(f"  [OK] {category}: {len(pairs)} 个物品对")
        return True
        
    except Exception as e:
        print(f"  [FAIL] {category}: 错误 - {e}")
        return False


def main():
    print("=" * 60)
    print("Step 3: 数据完整性校验")
    print("=" * 60)
    
    all_pass = True
    
    # 1. 校验单域训练CSV
    print("\n[1] 单域训练 CSV:")
    for category in TRAINING_CATEGORIES:
        csv_path = DATA_DIR / category / "5-core" / "train" / f"{category}_5_mixed.csv"
        if not verify_csv(csv_path, category):
            all_pass = False
    
    # 2. 校验混合训练CSV
    print("\n[2] AmazonMix-6 混合 CSV:")
    train_path = DATA_DIR / "AmazonMix-6" / "5-core" / "train" / "AmazonMix-6_5_mixed.csv"
    valid_path = DATA_DIR / "AmazonMix-6" / "5-core" / "valid" / "AmazonMix-6_5_mixed.csv"
    if not verify_csv(train_path, "Train"):
        all_pass = False
    if not verify_csv(valid_path, "Valid"):
        all_pass = False
    
    # 3. 校验 item_titles.txt
    print("\n[3] Item Titles:")
    if not verify_item_titles_txt():
        all_pass = False
    
    # 4. 校验 training_item_pairs
    print("\n[4] Training Item Pairs:")
    for category in TRAINING_CATEGORIES:
        if not verify_item_pairs(category):
            all_pass = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_pass:
        print("[SUCCESS] 所有数据校验通过!")
    else:
        print("[WARNING] 存在校验失败的文件,请检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
