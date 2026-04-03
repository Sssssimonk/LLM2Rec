# LLM2Rec 数据生成脚本

这个目录包含**4个独立的Python脚本**,用于生成LLM2Rec训练所需的所有数据文件。

## 📊 数据规模说明

论文使用的数据规模较小（如Video_Games: 153K交互），但Amazon原始数据集规模巨大（Video_Games: 736K训练交互）。为了匹配论文规模并减少处理时间，我们在**5-core过滤前先进行采样**。

## 🚀 完整数据处理流程

### Step 0: 原始数据采样（可选但推荐）

在5-core过滤前对原始数据采样，大幅减少数据量：

```bash
# 前置：先运行 notebooks/01_download_raw_data.ipynb 下载原始数据
python scripts/step0_sample_raw_data.py
```

**作用**：
- 基于用户随机采样，保留30%左右的活跃用户
- 原始数据量减少70%，处理时间大幅缩短
- 5-core过滤后数据量接近论文规模

**下一步**：运行 `notebooks/02_filter_and_split.ipynb` 进行5-core过滤和序列构建（需修改数据路径为`intermediate_sampled`）

### Step 1-3: 训练数据生成

在完成采样和5-core过滤后，按顺序执行：

```bash
# Step 1: 生成训练CSV文件
python scripts/step1_generate_csv.py

# Step 2: 生成IEM训练数据
python scripts/step2_generate_iem_data.py

# Step 3: 校验数据完整性
python scripts/step3_verify.py
```

## 📁 输出文件

### Step 0: 采样后的原始数据
- `intermediate_sampled/{Category}_interactions.parquet`
- `intermediate_sampled/{Category}_items.parquet`

### Step 1: 训练CSV
- 单域训练CSV: `{Category}/5-core/train/{Category}_5_mixed.csv`
- 混合训练CSV: `AmazonMix-6/5-core/train/AmazonMix-6_5_mixed.csv`
- 混合验证CSV: `AmazonMix-6/5-core/valid/AmazonMix-6_5_mixed.csv`

### Step 2: IEM数据
- 物品标题文本: `AmazonMix-6/5-core/info/item_titles.txt`
- 物品对数据: `{Category}/training_item_pairs_gap24.jsonl`

### Step 3: 数据校验
打印所有生成文件的统计信息和格式正确性

## 📋 前置条件

**Step 0** 需要：
- 运行 `notebooks/01_download_raw_data.ipynb` 下载原始数据到 `intermediate/`

**Step 1-3** 需要：
- 运行 `notebooks/02_filter_and_split.ipynb` 生成downstream评估文件：
  - `{Category}/5-core/downstream/train_data.txt`
  - `{Category}/5-core/downstream/item_titles.json`

## 💡 为什么需要Step 0采样？

| 数据集 | 原始训练交互 | 论文5-core后交互 | 采样率 |
|--------|--------------|------------------|--------|
| Video_Games | 736K | 153K | 30% |
| Movies_and_TV | 7.1M | 136K | 5% |
| Electronics | 1.4M | 198K | 20% |

**不采样的问题**：
- ❌ 处理时间过长（数小时）
- ❌ 内存占用过大
- ❌ 生成的数据量远超论文规模（可能影响模型训练）

**采样后的优势**：
- ✅ 处理时间缩短70%以上
- ✅ 内存占用大幅降低
- ✅ 最终数据量匹配论文规模
- ✅ 保持数据分布的代表性
