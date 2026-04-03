@echo off
chcp 65001
echo ============================================================
echo LLM2Rec 数据生成流水线
echo ============================================================
echo.

echo [Step 1/3] 生成训练CSV文件...
python scripts\step1_generate_csv.py
if errorlevel 1 (
    echo [ERROR] Step 1 失败
    pause
    exit /b 1
)
echo.

echo [Step 2/3] 生成IEM训练数据...
python scripts\step2_generate_iem_data.py
if errorlevel 1 (
    echo [ERROR] Step 2 失败
    pause
    exit /b 1
)
echo.

echo [Step 3/3] 数据校验...
python scripts\step3_verify.py
if errorlevel 1 (
    echo [ERROR] Step 3 失败
    pause
    exit /b 1
)
echo.

echo ============================================================
echo 所有步骤完成!
echo ============================================================
pause
