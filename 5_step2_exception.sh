#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medical_paper


# 设置文件夹路径



PYTHON_SCRIPT="step2/step2_excpetion_process.py"
# LLM="openai"
# LLM="deepseek"
LLM="qwq"


python "$PYTHON_SCRIPT" "$LLM" 

echo "All tasks completed!"




###    nohup bash 5_step2_exception.sh > logfile_2_exception_openAI.log 2>&1 &
###    nohup bash 5_step2_exception.sh > logfile_2_exception_deepseek.log 2>&1 &
###    nohup bash 5_step2_exception.sh > logfile_2_exception_qwq.log 2>&1 &


