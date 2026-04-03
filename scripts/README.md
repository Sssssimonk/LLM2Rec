# LLM2Rec 数据生成脚本

这个目录包含3个独立的Python脚本,用于生成LLM2Rec训练所需的所有数据文件。

## 使用方法

按顺序执行以下3个脚本:

```bash
# Step 1: 生成训练CSV文件
python scripts/step1_generate_csv.py

# Step 2: 生成IEM训练数据
python scripts/step2_generate_iem_data.py

# Step 3: 校验数据完整性
python scripts/step3_verify.py
```

## 输出文件

### Step 1: 训练CSV
- 单域训练CSV: `{Category}/5-core/train/{Category}_5_mixed.csv`
- 混合训练CSV: `AmazonMix-6/5-core/train/AmazonMix-6_5_mixed.csv`
- 混合验证CSV: `AmazonMix-6/5-core/valid/AmazonMix-6_5_mixed.csv`

### Step 2: IEM数据
- 物品标题文本: `AmazonMix-6/5-core/info/item_titles.txt`
- 物品对数据: `{Category}/training_item_pairs_gap24.jsonl`

### Step 3: 数据校验
打印所有生成文件的统计信息和格式正确性

## 前置条件

需要已有的downstream评估文件:
- `{Category}/5-core/downstream/data.txt`
- `{Category}/5-core/downstream/item_titles.json`

这些文件应该已经通过之前的notebook生成完成。
