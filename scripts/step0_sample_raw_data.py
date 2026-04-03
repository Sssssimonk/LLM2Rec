#!/usr/bin/env python3
"""
Step 0: 原始数据采样 - 匹配论文数据规模

问题：
- Amazon原始数据集规模巨大 (Video_Games: 736K训练交互)
- 论文中的数据规模较小 (Video_Games: 153K总交互)
- 完整处理耗时耗力，且超出论文需求

解决方案：
在5-core过滤之前，对原始数据进行采样，保留约30%的用户
这样可以：
1. 大幅减少处理时间 (原始数据量减少70%)
2. 匹配论文的数据规模
3. 保持数据分布的代表性

采样策略：
- 基于用户的随机采样 (user-based sampling)
- 每个领域保留30%的活跃用户
- 保留这些用户的所有交互记录
- 保证采样后仍能通过5-core过滤

数据对比：
| 数据集 | 原始训练交互 | 论文总交互 | 采样率 |
|--------|--------------|-----------|--------|
| Video_Games | 736K | 153K | ~30% |
| Movies_and_TV | 7.1M | 136K | ~5% |
| Electronics | 1.4M | 198K | ~20% |

使用方法：
1. 先运行 01_download_raw_data.ipynb 下载原始数据
2. 运行此脚本进行采样: python scripts/step0_sample_raw_data.py
3. 运行 02_filter_and_split.ipynb 进行5-core过滤
4. 运行 scripts/step1_generate_csv.py 生成训练CSV
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据目录
INTERMEDIATE_DIR = PROJECT_ROOT / 'llm2rec' / 'data' / 'intermediate'
SAMPLED_DIR = PROJECT_ROOT / 'llm2rec' / 'data' / 'intermediate_sampled'
SAMPLED_DIR.mkdir(parents=True, exist_ok=True)

# 6个训练领域 (AmazonMix-6)
TRAIN_CATEGORIES = [
    'Video_Games',
    'Arts_Crafts_and_Sewing', 
    'Movies_and_TV',
    'Home_and_Kitchen',
    'Electronics',
    'Tools_and_Home_Improvement'
]

# 论文中的数据规模 (5-core后)
PAPER_STATS = {
    'Video_Games': {'items': 9517, 'interactions': 153221},
    'Arts_Crafts_and_Sewing': {'items': 12454, 'interactions': 132566},
    'Movies_and_TV': {'items': 13190, 'interactions': 136471},
    'Home_and_Kitchen': {'items': 33478, 'interactions': 256001},
    'Electronics': {'items': 20150, 'interactions': 197984},
    'Tools_and_Home_Improvement': {'items': 19964, 'interactions': 159969},
}

# 每个领域的采样率 (基于原始数据规模估算)
SAMPLING_RATES = {
    'Video_Games': 0.30,              # 736K → 220K (5-core后约150K)
    'Arts_Crafts_and_Sewing': 0.40,   # 需要保留更多用户
    'Movies_and_TV': 0.05,            # 7.1M → 355K (5-core后约140K)
    'Home_and_Kitchen': 0.30,         # 2.5M → 750K (5-core后约260K)
    'Electronics': 0.20,              # 1.4M → 280K (5-core后约200K)
    'Tools_and_Home_Improvement': 0.15, # 6.0M → 900K (5-core后约160K)
}


def sample_interactions(df: pd.DataFrame, sampling_rate: float, seed: int = 42) -> pd.DataFrame:
    """
    基于用户的采样策略
    
    Args:
        df: 交互DataFrame (user_id, parent_asin, timestamp)
        sampling_rate: 用户采样率 (0-1)
        seed: 随机种子
    
    Returns:
        采样后的DataFrame
    """
    # 获取所有唯一用户
    all_users = df['user_id'].unique()
    n_total = len(all_users)
    
    # 计算要保留的用户数
    n_sample = int(n_total * sampling_rate)
    
    # 随机采样用户
    np.random.seed(seed)
    sampled_users = np.random.choice(all_users, size=n_sample, replace=False)
    
    # 保留这些用户的所有交互
    sampled_df = df[df['user_id'].isin(sampled_users)].copy()
    
    return sampled_df


def process_category(category: str):
    """处理单个类别的采样"""
    print(f"\n{'='*60}")
    print(f"处理类别: {category}")
    print(f"{'='*60}")
    
    # 读取原始交互数据
    interactions_file = INTERMEDIATE_DIR / f'{category}_interactions.parquet'
    items_file = INTERMEDIATE_DIR / f'{category}_items.parquet'
    
    if not interactions_file.exists():
        print(f"  ⚠️  未找到原始数据: {interactions_file}")
        print(f"  → 请先运行 notebooks/01_download_raw_data.ipynb")
        return
    
    # 加载数据
    print(f"  📂 加载原始数据...")
    interactions = pd.read_parquet(interactions_file)
    items = pd.read_parquet(items_file)
    
    print(f"  📊 原始数据规模:")
    print(f"     - 用户数: {interactions['user_id'].nunique():,}")
    print(f"     - 物品数: {interactions['parent_asin'].nunique():,}")
    print(f"     - 交互数: {len(interactions):,}")
    
    # 获取采样率
    sampling_rate = SAMPLING_RATES.get(category, 0.30)
    print(f"\n  🎲 采样率: {sampling_rate:.1%}")
    
    # 进行采样
    sampled_interactions = sample_interactions(interactions, sampling_rate)
    
    print(f"\n  ✂️  采样后数据规模:")
    print(f"     - 用户数: {sampled_interactions['user_id'].nunique():,}")
    print(f"     - 物品数: {sampled_interactions['parent_asin'].nunique():,}")
    print(f"     - 交互数: {len(sampled_interactions):,}")
    
    # 过滤items，只保留有交互的物品
    sampled_items_ids = sampled_interactions['parent_asin'].unique()
    sampled_items = items[items['parent_asin'].isin(sampled_items_ids)].copy()
    
    print(f"     - 有效物品元数据: {len(sampled_items):,}")
    
    # 论文对比
    if category in PAPER_STATS:
        paper_interactions = PAPER_STATS[category]['interactions']
        ratio = len(sampled_interactions) / paper_interactions
        print(f"\n  📈 与论文对比 (5-core前):")
        print(f"     - 论文 (5-core后): {paper_interactions:,} 交互")
        print(f"     - 采样 (5-core前): {len(sampled_interactions):,} 交互")
        print(f"     - 比例: {ratio:.2f}x (预期5-core后约为1.0-1.5x)")
    
    # 保存采样后的数据
    output_interactions = SAMPLED_DIR / f'{category}_interactions.parquet'
    output_items = SAMPLED_DIR / f'{category}_items.parquet'
    
    sampled_interactions.to_parquet(output_interactions, index=False)
    sampled_items.to_parquet(output_items, index=False)
    
    print(f"\n  ✅ 已保存:")
    print(f"     - {output_interactions}")
    print(f"     - {output_items}")


def main():
    """主函数"""
    print("="*70)
    print("Step 0: 原始数据采样 - 匹配论文数据规模")
    print("="*70)
    print("\n📝 目标: 在5-core过滤前采样，减少数据量到论文规模")
    print("\n💡 策略: 基于用户的随机采样，保留30%左右的活跃用户")
    print("   → 原始数据减少70%，但保持分布代表性")
    print("   → 5-core过滤后约匹配论文的数据量\n")
    
    # 检查原始数据目录
    if not INTERMEDIATE_DIR.exists():
        print(f"❌ 未找到中间数据目录: {INTERMEDIATE_DIR}")
        print(f"→ 请先运行 notebooks/01_download_raw_data.ipynb")
        return
    
    # 处理所有训练类别
    for category in TRAIN_CATEGORIES:
        try:
            process_category(category)
        except Exception as e:
            print(f"\n❌ 处理 {category} 时出错: {e}")
            continue
    
    print("\n" + "="*70)
    print("✅ 采样完成!")
    print("="*70)
    print(f"\n📁 采样后数据保存在: {SAMPLED_DIR}")
    print("\n🚀 下一步: 运行 notebooks/02_filter_and_split.ipynb 进行5-core过滤")
    print("   注意: 需要修改notebook中的数据路径为 'intermediate_sampled'")


if __name__ == '__main__':
    main()
