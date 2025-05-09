#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medical_paper






# 设置文件夹路径
abstract_title_path="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_extracted"
need_ids_dic="{}"  



PYTHON_SCRIPT="step2/step2_agent_openai_multi.py"
# PYTHON_SCRIPT="step2/step2_agent_deepseek_multi.py"
# PYTHON_SCRIPT="step2/step2_agent_qwq_multi.py"



for file in "$abstract_title_path"/*; do
    filename=$(basename "$file")
    id_part="${filename#SR_}"

    python "$PYTHON_SCRIPT" "$id_part" "$need_ids_dic" &
done

wait
echo "All tasks completed!"




###    nohup bash 4_step2_filter.sh > logfile_4_step2_filter_openai.log 2>&1 &
###    nohup bash 4_step2_filter.sh > logfile_4_step2_filter_deepseek.log 2>&1 &
###    nohup bash 4_step2_filter.sh > logfile_4_step2_filter_qwq.log 2>&1 &




