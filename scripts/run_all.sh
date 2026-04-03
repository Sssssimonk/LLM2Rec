#!/bin/bash

echo "============================================================"
echo "LLM2Rec 数据生成流水线"
echo "============================================================"
echo ""
echo "⚠️  前置步骤检查:"
echo "  1. 已运行 notebooks/01_download_raw_data.ipynb ?"
echo "  2. 已运行 scripts/step0_sample_raw_data.py (推荐) ?"
echo "  3. 已运行 notebooks/02_filter_and_split.ipynb ?"
echo ""
echo "如果未完成前置步骤,请先执行,否则按Enter继续..."
read -p ""

echo "[Step 1/3] 生成训练CSV文件..."
python scripts/step1_generate_csv.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Step 1 失败"
    exit 1
fi
echo ""

echo "[Step 2/3] 生成IEM训练数据..."
python scripts/step2_generate_iem_data.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Step 2 失败"
    exit 1
fi
echo ""

echo "[Step 3/3] 数据校验..."
python scripts/step3_verify.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Step 3 失败"
    exit 1
fi
echo ""

echo "============================================================"
echo "所有步骤完成!"
echo "============================================================"
