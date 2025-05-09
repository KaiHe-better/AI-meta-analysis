#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medical_paper


# 设置文件夹路径
abstract_title_path="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/abstract_title"
need_ids_dic="{}"  



# PYTHON_SCRIPT="step1/step1_agent_openai_multi.py"
# PYTHON_SCRIPT="step1/step1_deepseek_multi.py"
PYTHON_SCRIPT="step1/step1_agent_qwq_multi.py"



for file in "$abstract_title_path"/*; do
    python "$PYTHON_SCRIPT" "$file" "$need_ids_dic" &
done

wait
echo "All tasks completed!"




###    nohup bash run.sh > log_openAI.log 2>&1 &
###    nohup bash run.sh > log_deepseek.log 2>&1 &
###    nohup bash run.sh > log_qwq.log 2>&1 &


